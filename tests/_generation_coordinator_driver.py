"""Coordinator lifetime regressions, isolated behind an external test timeout."""
import sys
import threading
import os
import socket
import struct
import time
from types import SimpleNamespace
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client

import silk
from silk.simulation import generation as g


def expect_timeout(address, key):
    try:
        g._failure_rpc(address, key, 'abandon', ('expired-client',))
    except TimeoutError:
        return
    raise AssertionError('Expected the unserved client to expire')


def shutdown_control(mode):
    receiving = threading.Event()
    late_dispatched = threading.Event()
    original_recv = g._DeadlineConnection.recv_bytes
    original_abandon = g._abandon
    original_budget = g._SERVER_CONNECTION_TIMEOUT
    original_threading = g.threading
    received_frame = threading.Event()
    client = None

    def recv_bytes(self, maxlength=None):
        if (threading.current_thread().name == 'silk-generation-coordinator'
                and (maxlength is None or mode == 'stalled-auth')):
            receiving.set()
        frame = original_recv(self, maxlength)
        if threading.current_thread().name == 'silk-generation-coordinator' and maxlength is None:
            received_frame.set()
        return frame

    def abandon(ident):
        if ident == 'late-probe':
            late_dispatched.set()
        return original_abandon(ident)

    g._DeadlineConnection.recv_bytes = recv_bytes
    g._abandon = abandon
    g._SERVER_CONNECTION_TIMEOUT = 1.0
    if mode == 'during-shutdown':
        class StopThenSend(threading.Event):
            def set(self):
                super().set()
                client.send(('abandon', ('late-probe',)))

        # Replace only generation's Event factory, not Thread's internal events.
        g.threading = SimpleNamespace(Event=StopThenSend, Thread=threading.Thread)
    try:
        with g.worker_coordinator() as start:
            address, key = g._endpoint
            start()
            service, = [thread for thread in threading.enumerate()
                        if thread.name == 'silk-generation-coordinator']
            if mode == 'stalled-auth':
                client = socket.create_connection(address, timeout=2)
            else:
                client = Client(address, authkey=key)
                if mode == 'partial-frame':
                    os.write(client.fileno(), struct.pack('!i', 64) + b'x')
            assert receiving.wait(3), 'service never entered the blocking read'
            started = time.monotonic()
        assert time.monotonic() - started < 4
        assert not service.is_alive(), 'accepted connection outlived coordinator'
        assert g._endpoint is None
        if mode == 'during-shutdown':
            assert received_frame.is_set(), 'valid frame did not arrive during shutdown'
            assert not late_dispatched.is_set()
        if mode == 'late-dispatch':
            try:
                client.send(('abandon', ('late-probe',)))
            except (OSError, EOFError):
                pass
            assert not late_dispatched.is_set()
    finally:
        if client is not None:
            client.close()
        g._DeadlineConnection.recv_bytes = original_recv
        g._abandon = original_abandon
        g._SERVER_CONNECTION_TIMEOUT = original_budget
        g.threading = original_threading


def main():
    print(silk.__file__, flush=True)
    mode = sys.argv[1]
    if mode in ('idle', 'late-dispatch', 'partial-frame', 'stalled-auth', 'during-shutdown'):
        shutdown_control(mode)
        return
    with g.worker_coordinator() as start:
        address, key = g._endpoint
        if mode == 'delayed':
            expect_timeout(address, key)
            start()
        elif mode == 'in-progress':
            entered, release = threading.Event(), threading.Event()
            original = g._abandon
            replies = []

            def blocked(ident):
                if ident == 'active-request':
                    entered.set()
                    assert release.wait(8)
                return original(ident)

            def request():
                with Client(address, authkey=key) as connection:
                    connection.send(('abandon', ('active-request',)))
                    replies.append(connection.recv())

            g._abandon = blocked
            start()
            client = threading.Thread(target=request, daemon=True)
            client.start()
            try:
                assert entered.wait(3)
                expect_timeout(address, key)
            finally:
                release.set()
            client.join(timeout=3)
            assert not client.is_alive()
            assert replies == [(True, None)]
            g._abandon = original
        elif mode == 'stopped':
            start()
            service, = [thread for thread in threading.enumerate()
                        if thread.name == 'silk-generation-coordinator']
            with Client(address, authkey=key) as connection:
                connection.send(None)
            service.join(timeout=3)
            assert not service.is_alive()
            return
        elif mode == 'bad-key':
            start()
            try:
                with Client(address, authkey=b'wrong-key'):
                    raise AssertionError('Unauthenticated client was accepted')
            except AuthenticationError:
                pass
        elif mode in ('slow-request', 'slow-auth'):
            import multiprocessing.connection as connections
            start()
            original_answer = connections.answer_challenge

            def delayed_answer(connection, authkey):
                time.sleep(1.5)
                return original_answer(connection, authkey)

            if mode == 'slow-auth':
                connections.answer_challenge = delayed_answer
            try:
                with Client(address, authkey=key) as client:
                    if mode == 'slow-request':
                        time.sleep(1.5)
                    client.send(('abandon', ('paused-healthy-client',)))
                    assert client.recv() == (True, None)
            finally:
                connections.answer_challenge = original_answer
        else:
            raise AssertionError(mode)
        assert g._failure_rpc(address, key, 'abandon', ('healthy-client',)) == (True, None)
    assert g._endpoint is None


if __name__ == '__main__':
    main()
