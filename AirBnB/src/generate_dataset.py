import pandas as pd

def generate_dataset(df_master: pd.DataFrame,
                     listing_neighborhoods: list = None,
                     guest_country: list = None,
                     host_country: list = None,
                     room_type: list = None,
                     interval: str = 'hour',
                     look_back: int = 1):

  df = df_master.copy()
  
  if listing_neighborhoods is not None:
    mask = df['listing_neighborhood'].isin(listing_neighborhoods)
    df = df[mask].reset_index(drop = True)

  if guest_country is not None:
    mask = df['guest_country'].isin(guest_country)
    df = df[mask].reset_index(drop = True)
  
  if host_country is not None:
    mask = df['host_country'].isin(host_country)
    df = df[mask].reset_index(drop = True)
  
  if room_type is not None:
    mask = df['room_type'].isin(room_type)
    df = df[mask].reset_index(drop = True)
    
  # Convert timestamps to datetimes
  df.ts_interaction_first = pd.to_datetime(df.ts_interaction_first) # .dt.floor('D' if interval == 'day' else 'H')
  df.ts_reply_at_first = pd.to_datetime(df.ts_reply_at_first) # .dt.floor('D' if interval == 'day' else 'H')
  df.ts_accepted_at_first = pd.to_datetime(df.ts_accepted_at_first) # .dt.floor('D' if interval == 'day' else 'H')
  df.ts_booking_at = pd.to_datetime(df.ts_booking_at) # .dt.floor('D' if interval == 'day' else 'H')

  # Get min and max times
  max_time = min(max(df.ts_interaction_first), max(df.ts_reply_at_first), max(df.ts_accepted_at_first), max(df.ts_booking_at))
  min_time = max(min(df.ts_interaction_first), min(df.ts_reply_at_first), min(df.ts_accepted_at_first), min(df.ts_booking_at))

  time = pd.date_range(start = min_time, end = max_time, freq = '1D' if interval == 'day' else '1H')
  
  if interval == 'day':
    time = time.strftime('%Y-%m-%d')
  else:
    time = time.strftime('%Y-%m-%d %H:00')

  N = len(time)

  zeros_ = np.zeros((N-1,))
  nans_ = np.ones((N-1,)) * np.nan
  
  df_interval = pd.DataFrame({'date': time[1:],
                              'instant_book_rate': zeros_,
                              'accept_rate': zeros_,                              
                              'non_instant_book_rate': zeros_,
                              'accept_lag': nans_,
                              'book_lag': nans_,
                              'non_instant_book_lag': zeros_,                              
                              'total_book_rate': zeros_})
  
  for i in range(1, len(time)):
    min_time_i, max_time_i = time[i-1], time[i]

    interval_len = sum((df.ts_interaction_first > min_time_i) & (df.ts_interaction_first <= max_time_i))
    
    inquiry_idx_i = np.where(~df.ts_interaction_first.isnull() & (df.ts_interaction_first > min_time_i) & (df.ts_interaction_first <= max_time_i))[0]
    non_instant_inquiry_idx_i = np.where(~(df.contact_channel_first[inquiry_idx_i] != 'instant_book') & ~df.ts_interaction_first.isnull() & (df.ts_interaction_first > min_time_i) & (df.ts_interaction_first <= max_time_i))[0]

    reply_idx_i = np.where(~(df.contact_channel_first[inquiry_idx_i] != 'instant_book') & ~df.ts_reply_at_first.isnull() & (df.ts_reply_at_first > min_time_i) & (df.ts_reply_at_first <= max_time_i))[0]
    accepted_idx_i = np.where(~(df.contact_channel_first[inquiry_idx_i] != 'instant_book') & ~df.ts_accepted_at_first.isnull() & (df.ts_accepted_at_first > min_time_i) & (df.ts_accepted_at_first <= max_time_i))[0]
    booking_idx_i = np.where(~df.ts_booking_at.isnull() & (df.ts_booking_at > min_time_i) & (df.ts_booking_at <= max_time_i))[0]

    total_contact_me_i = sum(df.contact_channel_first[inquiry_idx_i] == 'contact_me')
    total_book_it_i = sum(df.contact_channel_first[inquiry_idx_i] == 'book_it')
    total_instant_book_i = sum(df.contact_channel_first[inquiry_idx_i] == 'instant_book')

    instant_book_idx_i = np.where(df.contact_channel_first[inquiry_idx_i] == 'instant_book')[0]
    non_instant_book_idx_i = np.where((df.contact_channel_first[inquiry_idx_i] != 'instant_book') & ~df.ts_booking_at.isnull())[0]

    total_inquiries_i = len(inquiry_idx_i)
    total_non_instant_inquiries_i = len(non_instant_inquiry_idx_i)

    total_replies_i = len(reply_idx_i)
    total_accepted_i = len(accepted_idx_i)
    total_bookings_i = len(booking_idx_i)
    total_non_instant_bookings_i = len(non_instant_book_idx_i)

    rate = lambda x,y: x/y if y != 0 else 0

    if total_inquiries_i > 0: 

        # instant_book_rate
        df_interval.loc[df_interval.index[i-1], 'instant_book_rate'] = rate(total_instant_book_i, total_bookings_i)

        # accept_rate
        df_interval.loc[df_interval.index[i-1], 'accept_rate'] = rate(total_accepted_i, total_non_instant_inquiries_i)

        # non_instant_book_rate
        df_interval.loc[df_interval.index[i-1], 'non_instant_book_rate'] = rate(total_non_instant_bookings_i, total_accepted_i)

        # accept_lag
        df_interval.loc[df_interval.index[i-1], 'accept_lag'] = (df.ts_accepted_at_first[non_instant_inquiry_idx_i] - df.ts_interaction_first[non_instant_inquiry_idx_i]).dropna().mean().seconds
        df_interval.loc[df_interval.index[i-1], 'accept_lag'] /= 3600 if interval == 'hour' else 3600*24

        # book_lag
        df_interval.loc[df_interval.index[i-1], 'book_lag'] = (df.ts_booking_at[non_instant_inquiry_idx_i] - df.ts_interaction_first[non_instant_inquiry_idx_i]).dropna().mean().seconds
        df_interval.loc[df_interval.index[i-1], 'book_lag'] /= 3600 if interval == 'hour' else 3600*24

        # non_instant_book_lag
        df_interval.loc[df_interval.index[i-1], 'non_instant_book_lag'] = (df.ts_booking_at[non_instant_inquiry_idx_i] - df.ts_accepted_at_first[non_instant_inquiry_idx_i]).dropna().mean().seconds
        df_interval.loc[df_interval.index[i-1], 'non_instant_book_lag'] /= 3600 if interval == 'hour' else 3600*24

        # total_book_rate
        df_interval.loc[df_interval.index[i-1], 'total_book_rate'] = rate(total_bookings_i, total_inquiries_i)

  return df_interval
