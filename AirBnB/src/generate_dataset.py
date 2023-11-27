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
  day = time.strftime('%A')
  
  if interval == 'day':
    time = time.strftime('%Y-%m-%d')
  else:
    time = time.strftime('%Y-%m-%d %H:00')

  N = len(time)

  zeros_ = np.zeros((N-1,))
  nans_ = np.ones((N-1,)) * np.nan
  
  df_interval = pd.DataFrame({'date': time[1:],
                              'day': day[1:],
                              'n_guests': zeros_,
                              'n_hosts': zeros_,
                              'n_listings': zeros_,
                              'n_inquiries': zeros_,
                              'n_replies': zeros_,
                              'n_accepted': zeros_,
                              'n_instant_book': zeros_,
                              'n_non_instant_book': zeros_,
                              'book_rate': zeros_,
                              'instant_book_rate': zeros_,
                              'accept_rate': zeros_,                              
                              'non_instant_book_rate': zeros_,
                              'accept_lag': nans_,
                              'non_instant_book_lag': zeros_})
  
  day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
  df_interval['day'] = pd.Categorical(df_interval['day'], categories = day_order, ordered = True)

  interaction_isnot_null = ~df.ts_interaction_first.isnull()
  is_instant_book = df.contact_channel_first == 'instant_book'
  isnot_instant_book = df.contact_channel_first != 'instant_book'

  for i in range(1, len(time)):

    # Start and end times of current time point (day or hour)
    min_time_i, max_time_i = time[i-1], time[i]

    interaction_in_interval = (df.ts_interaction_first > min_time_i) & (df.ts_interaction_first <= max_time_i) & ~df.ts_interaction_first.isnull()
    reply_in_interval = (df.ts_reply_at_first > min_time_i) & (df.ts_reply_at_first <= max_time_i) & ~df.ts_reply_at_first.isnull()
    accept_in_interval = (df.ts_accepted_at_first > min_time_i) & (df.ts_accepted_at_first <= max_time_i) & ~df.ts_accepted_at_first.isnull()
    book_in_interval = (df.ts_booking_at > min_time_i) & (df.ts_booking_at <= max_time_i) & ~df.ts_booking_at.isnull()
    
    inquiry_idx_i = np.where(interaction_isnot_null & interaction_in_interval)[0]
    non_instant_book_inquiry_idx_i = np.where(isnot_instant_book & interaction_in_interval)[0]

    replies_idx_i = np.where(isnot_instant_book & reply_in_interval)[0]
    accepted_idx_i = np.where(isnot_instant_book & accept_in_interval)[0]
    book_idx_i = np.where(book_in_interval)[0]
    instant_book_idx_i = np.where(is_instant_book & book_in_interval)[0]
    non_instant_book_idx_i = np.where(isnot_instant_book & book_in_interval)[0]

    contact_is_contact_me = df.contact_channel_first[inquiry_idx_i] == 'contact_me'
    contact_is_book_it = df.contact_channel_first[inquiry_idx_i] == 'book_it'
    contact_is_instant_book = df.contact_channel_first[inquiry_idx_i] == 'instant_book'

    n_contact_me_i = sum(contact_is_contact_me)
    n_book_it_i = sum(contact_is_book_it)
    n_instant_book_i = sum(contact_is_instant_book)

    n_inquiries_i = len(inquiry_idx_i)
    n_non_instant_inquiries_i = len(non_instant_book_inquiry_idx_i)

    unique_guests_i = set(df.id_guest_anon[inquiry_idx_i])
    unique_hosts_i = set(df.id_host_anon[inquiry_idx_i])
    unique_listings_i = set(df.id_listing_anon[inquiry_idx_i])

    n_guests_i = len(unique_guests_i)
    n_hosts_i = len(unique_hosts_i)
    n_listings_i = len(unique_listings_i)
    
    n_replies_i = len(replies_idx_i)
    n_accepted_i = len(accepted_idx_i)

    n_book_i = len(book_idx_i)
    n_instant_book_i = len(instant_book_idx_i)
    n_non_instant_book_i = len(non_instant_book_idx_i)

    t_accept_i = df.ts_accepted_at_first[non_instant_book_inquiry_idx_i]
    t_interaction_i = df.ts_interaction_first[non_instant_book_inquiry_idx_i]
    t_non_instant_book_i = df.ts_booking_at[non_instant_book_idx_i]

    rate = lambda x,y: x/y if y != 0 else 0
    
    if n_inquiries_i > 0: 

        # Total unique guests
        df_interval.loc[df_interval.index[i-1], 'n_guests'] = n_guests_i

        # Total unique hosts
        df_interval.loc[df_interval.index[i-1], 'n_hosts'] = n_hosts_i

        # Total unique listings
        df_interval.loc[df_interval.index[i-1], 'n_listings'] = n_listings_i

        # Total unique inquiries
        df_interval.loc[df_interval.index[i-1], 'n_inquiries'] = n_inquiries_i

        # Total unique replies
        df_interval.loc[df_interval.index[i-1], 'n_replies'] = n_replies_i

        # Total unique accepted
        df_interval.loc[df_interval.index[i-1], 'n_accepted'] = n_accepted_i

        # Total unique instant book
        df_interval.loc[df_interval.index[i-1], 'n_instant_book'] = n_instant_book_i

        # Total unique non-instant book
        df_interval.loc[df_interval.index[i-1], 'n_non_instant_book'] = n_non_instant_book_i

        # book_rate
        df_interval.loc[df_interval.index[i-1], 'book_rate'] = rate(n_book_i, n_inquiries_i)

        # instant_book_rate
        df_interval.loc[df_interval.index[i-1], 'instant_book_rate'] = rate(n_instant_book_i, n_book_i)

        # non_instant_book_rate
        df_interval.loc[df_interval.index[i-1], 'non_instant_book_rate'] = rate(n_non_instant_book_i, n_accepted_i)

        # accept_rate
        df_interval.loc[df_interval.index[i-1], 'accept_rate'] = rate(n_accepted_i, n_non_instant_inquiries_i)

        # accept_lag
        df_interval.loc[df_interval.index[i-1], 'accept_lag'] = (t_accept_i - t_interaction_i).dropna().mean().seconds
        df_interval.loc[df_interval.index[i-1], 'accept_lag'] /= 3600 if interval == 'hour' else 3600*24

        # non_instant_book_lag
        df_interval.loc[df_interval.index[i-1], 'non_instant_book_lag'] = (t_non_instant_book_i - t_accept_i).dropna().mean().seconds
        df_interval.loc[df_interval.index[i-1], 'non_instant_book_lag'] /= 3600 if interval == 'hour' else 3600*24

  return df_interval
