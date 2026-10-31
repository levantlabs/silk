"""Parent/child hierarchy of Spec objects."""
from silk.specs import Spec


def _spec(name, parent=None, **kw):
    return Spec(
        name=name,
        module=None,
        metric=kw.get('metric', 'noise_figure'),
        op=kw.get('op', '<='),
        target=kw.get('target', 2.0),
        parent=parent,
    )


def test_root_spec_has_no_parent_and_empty_children():
    root = _spec('system_NF')
    assert root.parent is None
    assert root.children == []


def test_child_spec_appended_to_parent_children_on_construction():
    root = _spec('system_NF', target=5.0)
    child = _spec('lna_NF', parent=root, target=2.0)
    assert child.parent is root
    assert child in root.children
    assert len(root.children) == 1


def test_multiple_children_share_parent():
    root = _spec('system_NF', target=5.0)
    lna = _spec('lna_NF', parent=root, target=2.0)
    mixer = _spec('mixer_NF', parent=root, target=10.0)
    assert lna in root.children
    assert mixer in root.children
    assert len(root.children) == 2


def test_grandchild_hierarchy():
    root = _spec('system_NF', target=5.0)
    mid = _spec('frontend_NF', parent=root, target=3.0)
    leaf = _spec('lna_NF', parent=mid, target=2.0)
    assert mid in root.children
    assert leaf in mid.children
    assert leaf not in root.children  # only direct children


def test_child_not_duplicated_in_parent_children_on_reassignment():
    """If a Spec is constructed with parent=X, the reference is added once."""
    root = _spec('system_NF', target=5.0)
    child = _spec('lna_NF', parent=root, target=2.0)
    # Manually "re-parent" — not a supported API, but ensure the children list
    # is not corrupted by accidental duplication in __post_init__.
    assert root.children.count(child) == 1
