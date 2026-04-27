r"""!-----------------------------------------------------------------------
!BOP
!
! !MODULE:  time --- keep control of time \label{sec:time}
!
! !INTERFACE:
!   MODULE time
!
! !DESCRIPTION:
!  This module provides a number of routines/functions and variables
!  related to the mode time in GOTM.
!  The basic concept used in this module is that time is expressed
!  as two integers --- one is the true Julian day and the other is
!  seconds since midnight. All calculations with time then become
!  very simple operations on integers.
!
! !USES:
!   IMPLICIT NONE
!
! !PUBLIC MEMBER FUNCTIONS:
!   public                              :: init_time, calendar_date
!   public                              :: julian_day, update_time
!   public                              :: read_time_string
!   public                              :: write_time_string
!   public                              :: time_diff
!   public                              :: sunrise_sunset
!   public                              :: in_time_interval
!
! !PUBLIC DATA MEMBERS:
!   character(len=19), public           :: timestr
!   character(len=19), public           :: start
!   character(len=19), public           :: stop
!   REALTYPE,          public           :: timestep
!
! Original author(s): Karsten Bolding & Hans Burchard
!
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!EOP
!-----------------------------------------------------------------------
"""

import math
from dataclasses import dataclass, field

__all__ = [
    "GotmTime",
    "calendar_date",
    "julian_day",
    "time_diff",
    "sunrise_sunset",
    "in_time_interval",
    "read_time_string",
    "write_time_string",
]

# Gregorian calendar reform boundary (for calendar_date)
_IGREG_CAL: int = 2299161
# Gregorian calendar reform boundary (for julian_day)
_IGREG_JUL: int = 15 + 31 * (10 + 12 * 1582)


def calendar_date(julian: int) -> tuple[int, int, int]:
    """
    Convert true Julian day to calendar date --- year, month and day.
    Based on a similar routine in Numerical Recipes.

    Args:
        julian: True Julian day number.

    Returns:
        (yyyy, mm, dd) calendar date.
    """
    if julian >= _IGREG_CAL:
        x = ((julian - 1867216) - 0.25) / 36524.25
        ja = julian + 1 + int(x) - int(0.25 * x)
    else:
        ja = julian

    jb = ja + 1524
    jc = int(6680 + ((jb - 2439870) - 122.1) / 365.25)
    jd = int(365 * jc + 0.25 * jc)
    je = int((jb - jd) / 30.6001)

    dd = jb - jd - int(30.6001 * je)
    mm = je - 1
    if mm > 12:
        mm -= 12
    yyyy = jc - 4715
    if mm > 2:
        yyyy -= 1
    if yyyy <= 0:
        yyyy -= 1

    return yyyy, mm, dd


def julian_day(yyyy: int, mm: int, dd: int) -> int:
    """
    Convert a calendar date to true Julian day.
    Based on a similar routine in Numerical Recipes.

    Args:
        yyyy: Year (use negative integers for BCE; 0 is skipped — 1 BCE is -1).
        mm:   Month (1–12).
        dd:   Day (1–31).

    Returns:
        True Julian day number.
    """
    jy = yyyy
    if jy < 0:
        jy += 1
    if mm > 2:
        jm = mm + 1
    else:
        jy -= 1
        jm = mm + 13
    jul = int(math.floor(365.25 * jy) + math.floor(30.6001 * jm) + dd + 1720995)
    if dd + 31 * (mm + 12 * yyyy) >= _IGREG_JUL:
        ja = int(0.01 * jy)
        jul = jul + 2 - ja + int(0.25 * ja)
    return jul


def time_diff(jul1: int, secs1: int, jul2: int, secs2: int) -> float:
    """
    Return the time difference between two dates in seconds.

    The dates are given as Julian day and seconds of that day.
    Returns (jul1, secs1) − (jul2, secs2) in seconds.
    """
    return 86400.0 * (jul1 - jul2) + float(secs1 - secs2)


def sunrise_sunset(latitude: float, declination: float) -> tuple[float, float]:
    """
    Return the times of sunrise and sunset.

    Args:
        latitude:    Geographic latitude in degrees.
        declination: Solar declination in degrees.

    Returns:
        (sunrise, sunset) in decimal hours (UTC).
    """
    omega = math.acos(
        -math.tan(latitude * math.pi / 180.0)
        * math.tan(declination * math.pi / 180.0)
    )
    hour = omega * 180.0 / math.pi / 15.0
    return 12.0 - hour, 12.0 + hour


def in_time_interval(j1: int, s1: int, j: int, s: int, j2: int, s2: int) -> bool:
    """
    Return True if (j, s) lies within the closed interval [(j1, s1), (j2, s2)].

    Times are expressed as (Julian day, seconds of that day).
    """
    before = (j < j1) or (j == j1 and s < s1)
    after = (j > j2) or (j == j2 and s > s2)
    return not before and not after


def read_time_string(timestr: str) -> tuple[int, int]:
    """
    Convert a time string to Julian day and seconds of that day.

    The format of the time string must be: 'yyyy-mm-dd hh:mm:ss'.

    Returns:
        (julian_day, seconds_of_day)
    """
    yy = int(timestr[0:4])
    mm = int(timestr[5:7])
    dd = int(timestr[8:10])
    hh = int(timestr[11:13])
    mn = int(timestr[14:16])
    ss = int(timestr[17:19])
    jul = julian_day(yy, mm, dd)
    secs = 3600 * hh + 60 * mn + ss
    return jul, secs


def write_time_string(jul: int, secs: int) -> str:
    """
    Format Julian day and seconds of that day to a time string.

    Output format: 'yyyy-mm-dd hh:mm:ss'.
    """
    hh = secs // 3600
    mn = (secs - hh * 3600) // 60
    ss = secs - 3600 * hh - 60 * mn
    yy, mm, dd = calendar_date(jul)
    return f"{yy:04d}-{mm:02d}-{dd:02d} {hh:02d}:{mn:02d}:{ss:02d}"


@dataclass
class GotmTime:
    """
    GOTM time-management state, mirroring the Fortran 'time' module.

    Time is represented as true Julian day + integer seconds since midnight.
    All public fields mirror the Fortran module-level variables.

    timefmt controls how start/stop/MaxN are resolved:
      1 — MaxN given directly; fake start date 2000-01-01 00:00:00
      2 — start and stop strings given; MaxN computed from duration
      3 — start string + MaxN given; stop computed
    """

    timestep: float = 3600.0
    timefmt: int = 2
    start: str = "2000-01-01 00:00:00"
    stop: str = "2001-01-01 00:00:00"

    # Populated by init_time() / update_time()
    timestr: str = field(init=False, default="")
    julianday: int = field(init=False, default=0)
    secondsofday: int = field(init=False, default=0)
    fsecs: float = field(init=False, default=0.0)
    fsecondsofday: float = field(init=False, default=0.0)
    simtime: float = field(init=False, default=0.0)
    yearday: int = field(init=False, default=0)

    MinN: int = field(init=False, default=1)
    MaxN: int = field(init=False, default=0)
    jul1: int = field(init=False, default=0)
    secs1: int = field(init=False, default=0)
    jul2: int = field(init=False, default=0)
    secs2: int = field(init=False, default=0)

    _jul0: int = field(init=False, default=0, repr=False)
    _secs0: int = field(init=False, default=0, repr=False)
    _has_real_time: bool = field(init=False, default=True, repr=False)

    def init_time(self) -> None:
        """
        Initialise the time system.

        Resolves start/stop/MaxN according to timefmt, then calls
        update_time(0) to set all state for the initial instant.
        Mirrors init_time() from time.F90.
        """
        self.MinN = 1

        if self.timefmt == 1:
            self._has_real_time = False
            self.start = "2000-01-01 00:00:00"
            self.jul1, self.secs1 = read_time_string(self.start)
        elif self.timefmt == 2:
            self._has_real_time = True
            self.jul1, self.secs1 = read_time_string(self.start)
            self.jul2, self.secs2 = read_time_string(self.stop)
            nsecs = int(time_diff(self.jul2, self.secs2, self.jul1, self.secs1))
            self.MaxN = round(nsecs / self.timestep)
        elif self.timefmt == 3:
            self._has_real_time = True
            self.jul1, self.secs1 = read_time_string(self.start)
            nsecs = round(self.MaxN * self.timestep) + self.secs1
            ndays = nsecs // 86400
            self.jul2 = self.jul1 + ndays
            self.secs2 = nsecs % 86400
            self.stop = write_time_string(self.jul2, self.secs2)
        else:
            raise ValueError(f"Invalid timefmt: {self.timefmt}")

        self._jul0 = self.jul1
        self._secs0 = self.secs1
        self.update_time(0)
        self.simtime = self.timestep * (self.MaxN - self.MinN + 1)

    def update_time(self, n: int) -> None:
        """
        Update all time-tracking variables for time step n.

        Args:
            n: Time step index (0 = start of simulation).

        Mirrors update_time(n) from time.F90.
        """
        nsecs = round(n * self.timestep) + self._secs0
        self.fsecs = n * self.timestep + self._secs0
        self.julianday = self._jul0 + nsecs // 86400
        self.secondsofday = nsecs % 86400
        self.fsecondsofday = self.fsecs % 86400.0
        yyyy, _mm, _dd = calendar_date(self.julianday)
        jd_firstjan = julian_day(yyyy, 1, 1)
        self.yearday = self.julianday - jd_firstjan + 1
        self.timestr = write_time_string(self.julianday, self.secondsofday)
