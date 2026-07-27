"""pose3d.fit — global multi-view SMPL-X fitting (SMPLify-X based).

Optional fusion backend (`--fusion global_fit`). SCOPE/SCAFFOLD only — the
fitting loop + losses are TODO pending cali re-record + VPoser. The default
triangulate pipeline is untouched. See docs/MULTIVIEW_FITTING_PLAN.md.
"""
from .multiview_fitter import MultiViewFitter  # noqa: F401
