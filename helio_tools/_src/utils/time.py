from datetime import datetime
from dateutil.relativedelta import relativedelta


DATETIME_FORMATS = {"solo": "%Y-%m-%d %H:%M", "sodo": "%Y-%m-%d"}


def check_datetime_format(datetime_str: str, sensor: str = "sodo") -> datetime:
    try:
        datetime_obj = datetime.strptime(datetime_str, DATETIME_FORMATS[sensor])
        return datetime_obj
    except ValueError:
        msg = f"Datetime string format is incorrect for sensor: {sensor}."
        msg += f"\nInput: {datetime_str}"
        raise ValueError(msg)


def get_num_months(start_date: datetime, end_date: datetime) -> int:
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


def get_month_dates(start_date: datetime, num_months: int) -> list:
    return [start_date + i * relativedelta(months=1) for i in range(num_months)]
