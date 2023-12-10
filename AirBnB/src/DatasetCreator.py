def ts_span(ts1, ts2, unit = 'hour'):
    
    ts1 = pd.to_datetime(ts1) if not isinstance(ts1, pd.Timestamp) else ts1
    ts2 = pd.to_datetime(ts2) if not isinstance(ts2, pd.Timestamp) else ts2

    if unit == 'hour':
        scale = 3600
    elif unit == 'day':
        scale = 3600*24

    span = (ts2 - ts1).dt.total_seconds()/scale

    return span
