"""Tests that silk.layout exposes its full public API."""


class TestLayoutPackageExports:

    def test_routing_model(self):
        from silk.layout import RoutingModel
        assert RoutingModel is not None

    def test_half_circuit(self):
        from silk.layout import HalfCircuit
        assert HalfCircuit is not None

    def test_layout_inference_error(self):
        from silk.layout import LayoutInferenceError
        assert LayoutInferenceError is not None

    def test_layout_estimate(self):
        from silk.layout import LayoutEstimate
        assert LayoutEstimate is not None

    def test_net_layout(self):
        from silk.layout import NetLayout
        assert NetLayout is not None

    def test_device_layout(self):
        from silk.layout import DeviceLayout
        assert DeviceLayout is not None

    def test_layout_warning(self):
        from silk.layout import LayoutWarning
        assert LayoutWarning is not None

    def test_estimate_layout(self):
        from silk.layout import estimate_layout
        assert callable(estimate_layout)

    def test_capture_layout(self):
        from silk.layout import capture_layout
        assert callable(capture_layout)

    def test_inject_parasitics(self):
        from silk.layout import inject_parasitics
        assert callable(inject_parasitics)


class TestSilkTopLevelLayoutExports:

    def test_routing_model_from_silk(self):
        import silk
        assert hasattr(silk, 'RoutingModel')

    def test_half_circuit_from_silk(self):
        import silk
        assert hasattr(silk, 'HalfCircuit')

    def test_estimate_layout_from_silk(self):
        import silk
        assert hasattr(silk, 'estimate_layout')

    def test_capture_layout_from_silk(self):
        import silk
        assert hasattr(silk, 'capture_layout')

    def test_inject_parasitics_from_silk(self):
        import silk
        assert hasattr(silk, 'inject_parasitics')

    def test_layout_estimate_from_silk(self):
        import silk
        assert hasattr(silk, 'LayoutEstimate')

    def test_net_layout_from_silk(self):
        import silk
        assert hasattr(silk, 'NetLayout')

    def test_device_layout_from_silk(self):
        import silk
        assert hasattr(silk, 'DeviceLayout')

    def test_layout_warning_from_silk(self):
        import silk
        assert hasattr(silk, 'LayoutWarning')

    def test_layout_inference_error_from_silk(self):
        import silk
        assert hasattr(silk, 'LayoutInferenceError')
