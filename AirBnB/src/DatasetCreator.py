def ts_span(ts1, ts2, unit = 'hour'):
    
    ts1 = pd.to_datetime(ts1) if not isinstance(ts1, pd.Timestamp) else ts1
    ts2 = pd.to_datetime(ts2) if not isinstance(ts2, pd.Timestamp) else ts2

    if unit == 'hour':
        scale = 3600
    elif unit == 'day':
        scale = 3600*24

    span = (ts2 - ts1).dt.total_seconds()/scale

    return span

def ts_extract(ts, unit = 'hour'):

    if unit == 'hour':
        t = ts.dt.hour
    elif unit == 'day':
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        t = pd.Categorical(ts.dt.strftime('%A'), categories = order, ordered = True)        
    elif unit == 'month':
        order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October' , 'November', 'December']
        t = pd.Categorical(ts.dt.strftime('%B'), categories = order, ordered = True)
    elif unit == 'year':
        t = ts.dt.year
    else:
        raise ValueError(f"Invalid unit: {unit}")
        
    return t

def create_dataset(df_master):
    
    df = df_master.copy()

    inquiry_date = pd.to_datetime(df['ts_interaction_first'])
    reply_date = pd.to_datetime(df['ts_reply_at_first'])
    accept_date = pd.to_datetime(df['ts_accepted_at_first'])
    book_date = pd.to_datetime(df['ts_booking_at'])

    checkin_date = pd.to_datetime(df['ds_checkin_first'])
    checkout_date = pd.to_datetime(df['ds_checkout_first'])

    dataset = pd.DataFrame()

    dataset['inquiry_date'] = inquiry_date
    dataset['reply_date'] = reply_date
    dataset['accept_date'] = accept_date
    dataset['book_date'] = book_date

    # year, month, day, and hour
    for unit in ['year', 'month', 'day', 'hour']:
        dataset[f"inquiry_{unit}"] = ts_extract(inquiry_date, unit)
        dataset[f"reply_{unit}"] = ts_extract(reply_date, unit)
        dataset[f"accept_{unit}"] = ts_extract(accept_date, unit)
        dataset[f"book_{unit}"] = ts_extract(book_date, unit)

    # inquiry type
    dataset['inquiry_type'] = df_master['contact_channel_first']

    # replied?, accepted?, booked?
    dataset['replied'] = (~reply_date.isna()).astype(int)
    dataset['accepted'] = (~accept_date.isna()).astype(int)
    dataset['booked'] = (~book_date.isna()).astype(int)

    # lags
    dataset['reply_lag'] = ts_span(inquiry_date, reply_date, 'hour')
    dataset['accept_lag'] = ts_span(inquiry_date, accept_date, 'hour')
    dataset['book_lag'] = ts_span(inquiry_date, book_date, 'hour')

    # stay duration
    dataset['stay_duration'] = ts_span(checkin_date, checkout_date, 'day')

    # past booker
    dataset['past_booker'] = (df['guest_user_stage_first'] == 'past_booker').astype(int)

    # id
    dataset['guest_id'] = df_master['id_guest_anon']
    dataset['host_id'] = df['id_host_anon']

    # country
    dataset['guest_country'] = df['guest_country']
    dataset['host_country'] = df['host_country']

    # number of guests
    dataset['n_guests']  = df['m_guests']

    # number of  guest-host interactions
    dataset['n_interactions']  = df_master['m_interactions']

    # room type
    dataset['room_type']  = df['room_type']

    # number of reviews
    dataset['n_reviews']  = df['total_reviews']

    # Length of Guest's first message to host
    dataset['guest_msg_length'] = df['m_first_message_length_in_characters']

    # Number of words in user's "about me"
    dataset['n_guest_about_me_words'] = df['words_in_guest_profile']
    dataset['n_host_about_me_words'] = df['words_in_host_profile']
    
    return dataset
