"""
Time utilities for MCT.
"""
from datetime import datetime, timezone, timedelta


# Vietnam timezone
VN_TZ = timezone(timedelta(hours=7))


def get_vn_time():
    """Get current time in Vietnam timezone"""
    return datetime.now(VN_TZ)
