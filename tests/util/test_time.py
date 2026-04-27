"""Tests for pygotm.util.time — GOTM time management module."""

import math

import pytest

from pygotm.util.time import (
    GotmTime,
    calendar_date,
    in_time_interval,
    julian_day,
    read_time_string,
    sunrise_sunset,
    time_diff,
    write_time_string,
)


# ---------------------------------------------------------------------------
# Julian day / calendar date roundtrip
# ---------------------------------------------------------------------------


class TestJulianCalendarRoundtrip:
    """julian_day and calendar_date must be exact inverses."""

    @pytest.mark.parametrize(
        "yyyy, mm, dd",
        [
            (2000, 1, 1),
            (2000, 12, 31),
            (1582, 10, 15),  # first Gregorian day
            (1582, 10, 4),  # last Julian day
            (1970, 1, 1),  # Unix epoch
            (2024, 2, 29),  # leap day
            (1900, 3, 1),
            (2100, 2, 28),
            (-1, 1, 1),  # 2 BCE (Fortran convention: no year 0)
        ],
    )
    def test_roundtrip(self, yyyy: int, mm: int, dd: int) -> None:
        jul = julian_day(yyyy, mm, dd)
        y2, m2, d2 = calendar_date(jul)
        assert (y2, m2, d2) == (yyyy, mm, dd)

    def test_known_j2000(self) -> None:
        # J2000.0 epoch is 2000 January 1.5 → Julian day 2451545
        assert julian_day(2000, 1, 1) == 2451545

    def test_unix_epoch_julian(self) -> None:
        # 1970-01-01 → Julian day 2440588
        assert julian_day(1970, 1, 1) == 2440588

    def test_sequential_days(self) -> None:
        # Julian days must be consecutive for consecutive calendar dates
        jul1 = julian_day(2023, 2, 28)
        jul2 = julian_day(2023, 3, 1)
        assert jul2 - jul1 == 1

    def test_leap_year_day_count(self) -> None:
        # 2024 is a leap year: 366 days
        first = julian_day(2024, 1, 1)
        last = julian_day(2024, 12, 31)
        assert last - first == 365  # 366 days = 365 intervals


# ---------------------------------------------------------------------------
# Time string I/O
# ---------------------------------------------------------------------------


class TestTimeStringIO:
    @pytest.mark.parametrize(
        "timestr",
        [
            "2000-01-01 00:00:00",
            "2023-06-15 12:30:45",
            "1999-12-31 23:59:59",
            "2024-02-29 06:00:00",
        ],
    )
    def test_roundtrip(self, timestr: str) -> None:
        jul, secs = read_time_string(timestr)
        out = write_time_string(jul, secs)
        assert out == timestr

    def test_midnight(self) -> None:
        jul, secs = read_time_string("2000-01-01 00:00:00")
        assert secs == 0

    def test_end_of_day(self) -> None:
        jul, secs = read_time_string("2000-01-01 23:59:59")
        assert secs == 86399

    def test_noon(self) -> None:
        _, secs = read_time_string("2000-01-01 12:00:00")
        assert secs == 43200

    def test_write_zero_padded(self) -> None:
        jul = julian_day(2000, 3, 5)
        s = write_time_string(jul, 60)  # 00:01:00
        assert s == "2000-03-05 00:01:00"


# ---------------------------------------------------------------------------
# time_diff
# ---------------------------------------------------------------------------


class TestTimeDiff:
    def test_same_day_positive(self) -> None:
        jul = julian_day(2000, 1, 1)
        assert time_diff(jul, 3600, jul, 0) == pytest.approx(3600.0)

    def test_same_day_negative(self) -> None:
        jul = julian_day(2000, 1, 1)
        assert time_diff(jul, 0, jul, 3600) == pytest.approx(-3600.0)

    def test_one_full_day(self) -> None:
        j1 = julian_day(2000, 1, 2)
        j2 = julian_day(2000, 1, 1)
        assert time_diff(j1, 0, j2, 0) == pytest.approx(86400.0)

    def test_identical_dates(self) -> None:
        jul = julian_day(2000, 6, 1)
        assert time_diff(jul, 1234, jul, 1234) == 0.0

    def test_cross_midnight(self) -> None:
        j1 = julian_day(2000, 1, 2)
        j2 = julian_day(2000, 1, 1)
        assert time_diff(j1, 3600, j2, 82800) == pytest.approx(3600.0 - 82800.0 + 86400.0)


# ---------------------------------------------------------------------------
# sunrise_sunset
# ---------------------------------------------------------------------------


class TestSunriseSunset:
    def test_equator_equinox(self) -> None:
        # At equator (lat=0) and equinox (declination=0), omega=pi/2
        # hour = 90/15 = 6 → sunrise=6, sunset=18
        sr, ss = sunrise_sunset(0.0, 0.0)
        assert sr == pytest.approx(6.0, abs=1e-10)
        assert ss == pytest.approx(18.0, abs=1e-10)

    def test_symmetry(self) -> None:
        sr, ss = sunrise_sunset(45.0, 10.0)
        assert ss == pytest.approx(24.0 - sr, abs=1e-10)

    def test_positive_declination_longer_day(self) -> None:
        # At positive latitude + positive declination → day > 12 hours
        sr1, ss1 = sunrise_sunset(45.0, 0.0)
        sr2, ss2 = sunrise_sunset(45.0, 15.0)
        assert (ss2 - sr2) > (ss1 - sr1)

    def test_polar_day_raises(self) -> None:
        # At pole with max declination, cos argument > 1 → math domain error
        with pytest.raises((ValueError, Exception)):
            sunrise_sunset(90.0, 23.5)

    def test_output_range(self) -> None:
        sr, ss = sunrise_sunset(30.0, 5.0)
        assert 0.0 < sr < 12.0
        assert 12.0 < ss < 24.0


# ---------------------------------------------------------------------------
# in_time_interval
# ---------------------------------------------------------------------------


class TestInTimeInterval:
    def setup_method(self) -> None:
        self.j1 = julian_day(2000, 1, 1)
        self.s1 = 0
        self.j2 = julian_day(2000, 1, 2)
        self.s2 = 0

    def test_start_is_inside(self) -> None:
        assert in_time_interval(self.j1, self.s1, self.j1, self.s1, self.j2, self.s2)

    def test_end_is_inside(self) -> None:
        assert in_time_interval(self.j1, self.s1, self.j2, self.s2, self.j2, self.s2)

    def test_midpoint_is_inside(self) -> None:
        assert in_time_interval(self.j1, self.s1, self.j1, 43200, self.j2, self.s2)

    def test_before_interval(self) -> None:
        j_before = self.j1 - 1
        assert not in_time_interval(self.j1, self.s1, j_before, 0, self.j2, self.s2)

    def test_after_interval(self) -> None:
        j_after = self.j2 + 1
        assert not in_time_interval(self.j1, self.s1, j_after, 0, self.j2, self.s2)

    def test_same_julian_second_boundary(self) -> None:
        # Exactly one second before start
        assert not in_time_interval(self.j1, 100, self.j1, 99, self.j2, self.s2)

    def test_same_julian_second_inside(self) -> None:
        assert in_time_interval(self.j1, 100, self.j1, 100, self.j2, self.s2)


# ---------------------------------------------------------------------------
# GotmTime — timefmt 2 (start + stop)
# ---------------------------------------------------------------------------


class TestGotmTimeFmt2:
    def setup_method(self) -> None:
        self.gt = GotmTime(
            timestep=3600.0,
            timefmt=2,
            start="2000-01-01 00:00:00",
            stop="2000-01-02 00:00:00",
        )
        self.gt.init_time()

    def test_maxn_correct(self) -> None:
        # 24 hours / 3600s per step = 24 steps
        assert self.gt.MaxN == 24

    def test_minn_is_one(self) -> None:
        assert self.gt.MinN == 1

    def test_initial_timestr(self) -> None:
        assert self.gt.timestr == "2000-01-01 00:00:00"

    def test_initial_secondsofday(self) -> None:
        assert self.gt.secondsofday == 0

    def test_initial_yearday(self) -> None:
        assert self.gt.yearday == 1

    def test_update_one_step(self) -> None:
        self.gt.update_time(1)
        assert self.gt.secondsofday == 3600
        assert self.gt.timestr == "2000-01-01 01:00:00"

    def test_update_to_end(self) -> None:
        self.gt.update_time(24)
        assert self.gt.timestr == "2000-01-02 00:00:00"
        assert self.gt.secondsofday == 0

    def test_yearday_advances(self) -> None:
        self.gt.update_time(24)
        assert self.gt.yearday == 2

    def test_fsecs_monotone(self) -> None:
        prev = self.gt.fsecs
        for n in range(1, 25):
            self.gt.update_time(n)
            assert self.gt.fsecs > prev
            prev = self.gt.fsecs

    def test_julianday_advances(self) -> None:
        j0 = self.gt.julianday
        self.gt.update_time(24)
        assert self.gt.julianday == j0 + 1

    def test_simtime(self) -> None:
        # simtime = timestep * (MaxN - MinN + 1) = 3600 * 24 = 86400
        assert self.gt.simtime == pytest.approx(86400.0)


# ---------------------------------------------------------------------------
# GotmTime — timefmt 1 (MaxN given, fake start)
# ---------------------------------------------------------------------------


class TestGotmTimeFmt1:
    def setup_method(self) -> None:
        self.gt = GotmTime(timestep=1800.0, timefmt=1)
        self.gt.MaxN = 48
        self.gt.init_time()

    def test_fake_start(self) -> None:
        assert self.gt.start == "2000-01-01 00:00:00"

    def test_maxn_preserved(self) -> None:
        assert self.gt.MaxN == 48

    def test_initial_timestr(self) -> None:
        assert self.gt.timestr == "2000-01-01 00:00:00"

    def test_final_timestr(self) -> None:
        self.gt.update_time(48)
        assert self.gt.timestr == "2000-01-02 00:00:00"


# ---------------------------------------------------------------------------
# GotmTime — timefmt 3 (start + MaxN, compute stop)
# ---------------------------------------------------------------------------


class TestGotmTimeFmt3:
    def setup_method(self) -> None:
        self.gt = GotmTime(
            timestep=3600.0,
            timefmt=3,
            start="2000-01-01 00:00:00",
        )
        self.gt.MaxN = 24
        self.gt.init_time()

    def test_stop_computed(self) -> None:
        assert self.gt.stop == "2000-01-02 00:00:00"

    def test_jul2_correct(self) -> None:
        j_expected = julian_day(2000, 1, 2)
        assert self.gt.jul2 == j_expected

    def test_secs2_zero(self) -> None:
        assert self.gt.secs2 == 0


# ---------------------------------------------------------------------------
# GotmTime — invalid timefmt
# ---------------------------------------------------------------------------


def test_invalid_timefmt_raises() -> None:
    gt = GotmTime(timefmt=99)
    with pytest.raises(ValueError):
        gt.init_time()


# ---------------------------------------------------------------------------
# NaN / Inf guard
# ---------------------------------------------------------------------------


class TestNanInfGuard:
    def test_calendar_date_no_nan(self) -> None:
        for julian in [1, 2440588, 2451545, 2299161, 2299160]:
            yyyy, mm, dd = calendar_date(julian)
            assert all(isinstance(v, int) for v in (yyyy, mm, dd))

    def test_time_diff_no_nan(self) -> None:
        j = julian_day(2000, 6, 1)
        diff = time_diff(j, 3600, j, 0)
        assert math.isfinite(diff)

    def test_sunrise_sunset_no_nan(self) -> None:
        sr, ss = sunrise_sunset(0.0, 0.0)
        assert math.isfinite(sr) and math.isfinite(ss)

    def test_gotmtime_fsecs_finite(self) -> None:
        gt = GotmTime(
            timestep=3600.0,
            timefmt=2,
            start="2000-01-01 00:00:00",
            stop="2000-01-02 00:00:00",
        )
        gt.init_time()
        for n in range(25):
            gt.update_time(n)
            assert math.isfinite(gt.fsecs)
            assert math.isfinite(gt.fsecondsofday)
