import pandas as pd

def generate_dataset(df_master: pd.DataFrame,
                     listing_neighborhoods: list = None,
                     guest_country: list = None,
                     host_country: list = None,
                     interval: str = 'hour',
                     look_back: int = 1):

  df = df_master.copy()

  if listing_neighborhoods is not None:
    mask = df['listing_neighborhood'].isin(listing_neighborhoods)
    df = df[mask].reset_index()

  if guest_country is not None:
    mask = df['guest_country'].isin(guest_country)
    df = df[mask].reset_index()

  if host_country is not None:
    mask = df['host_country'].isin(host_country)
    df = df[mask].reset_index()

  # Convert timestamps to datetimes
  df.ts_interaction_first = pd.to_datetime(df.ts_interaction_first) # .dt.floor('D' if interval == 'day' else 'H')
  df.ts_reply_at_first = pd.to_datetime(df.ts_reply_at_first) # .dt.floor('D' if interval == 'day' else 'H')
  df.ts_accepted_at_first = pd.to_datetime(df.ts_accepted_at_first) # .dt.floor('D' if interval == 'day' else 'H')
  df.ts_booking_at = pd.to_datetime(df.ts_booking_at) # .dt.floor('D' if interval == 'day' else 'H')

  # Get min and max times
  max_time = min(max(df.ts_interaction_first), max(df.ts_reply_at_first), max(df.ts_accepted_at_first), max(df.ts_booking_at))
  min_time = max(min(df.ts_interaction_first), min(df.ts_reply_at_first), min(df.ts_accepted_at_first), min(df.ts_booking_at))

  time = pd.date_range(start = min_time, end = max_time, freq = '1D' if interval == 'day' else '1H')

  N = len(time)

  zeros_ = np.zeros((N-1,))
  nans_ = np.ones((N-1,)) * np.nan

  df_interval = pd.DataFrame({# inquires
                              'total_inquiries': zeros_,
                              'total_home_inquiries': zeros_,
                              'total_shared_inquiries': zeros_,
                              'total_private_inquiries': zeros_,
                              'guest_rate': zeros_,
                              'guest_return_rate': zeros_,
                              'host_return_rate': zeros_,
                              'interaction_rate': zeros_,
                              'message_rate': zeros_,
                              'total_contact_me': zeros_,
                              'total_book_it': zeros_,
                              'total_instant_book': zeros_,
                              'pct_contact_me': zeros_,
                              'pct_book_it': zeros_,
                              'pct_instant_book': zeros_,
                              'pct_past_booker': zeros_,
                              'pct_new_user': zeros_,
                              # replies
                              'total_replies': zeros_,
                              'reply_rate': zeros_,
                              'avg_reply_time': nans_,
                              # accepted
                              'total_accepted': zeros_,
                              'accepted_rate': zeros_,
                              'avg_accepted_time': nans_,
                              # booking
                              'total_booking': zeros_,
                              'booking_rate': zeros_,
                              'avg_booking_time': nans_},
                              index = time[1:])

  inquiry_idx_I = []

  for i in range(1, len(time)):
    min_time_i, max_time_i = time[i-1], time[i]

    interval_len = sum((df.ts_interaction_first > min_time_i) & (df.ts_interaction_first <= max_time_i))

    inquiry_idx_i = np.where(~df.ts_interaction_first.isnull() & (df.ts_interaction_first > min_time_i) & (df.ts_interaction_first <= max_time_i))[0]
    reply_idx_i = np.where(~df.ts_reply_at_first.isnull() & (df.ts_reply_at_first > min_time_i) & (df.ts_reply_at_first <= max_time_i))[0]
    accepted_idx_i = np.where(~df.ts_accepted_at_first.isnull() & (df.ts_accepted_at_first > min_time_i) & (df.ts_accepted_at_first <= max_time_i))[0]
    booking_idx_i = np.where(~df.ts_booking_at.isnull() & (df.ts_booking_at > min_time_i) & (df.ts_booking_at <= max_time_i))[0]

    inquiry_idx_I.append(inquiry_idx_i)

    # number of inquiries
    df_interval.total_inquiries[i-1] = len(inquiry_idx_i)

    df_interval.total_home_inquiries[i-1] = sum(df.room_type[inquiry_idx_i] == 'Entire home/apt')
    df_interval.total_shared_inquiries[i-1] = sum(df.room_type[inquiry_idx_i] == 'Shared room')
    df_interval.total_private_inquiries[i-1] = sum(df.room_type[inquiry_idx_i] == 'Private room')

    # Guests per inquiry
    df_interval.guest_rate[i-1] = df.m_guests[inquiry_idx_i].sum() / df_interval.total_inquiries[i-1]

    if len(inquiry_idx_I) > 1:
      inquiry_idx_prev = np.concatenate(inquiry_idx_I[-(look_back+1):-1])

      total_guest_repeat = len(set(df.id_guest_anon[inquiry_idx_prev]) & set(df.id_guest_anon[inquiry_idx_i]))
      df_interval.guest_return_rate[i-1] = total_guest_repeat / len(np.unique(df.id_guest_anon[inquiry_idx_i]))

      total_host_repeat = len(set(df.id_host_anon[inquiry_idx_prev]) & set(df.id_host_anon[inquiry_idx_i]))
      df_interval.host_return_rate[i-1] = total_host_repeat / len(np.unique(df.id_host_anon[inquiry_idx_i]))

    # Interaction rate
    df_interval.interaction_rate[i-1] = np.sum(df.m_interactions[inquiry_idx_i]) / df_interval.total_inquiries[i-1]

    # Message length by inquiry
    df_interval.message_rate[i-1] = df.m_first_message_length_in_characters[inquiry_idx_i].sum() / df_interval.total_inquiries[i-1]

    # number of 'contact me', 'book it', and 'instant book' inquires
    df_interval.total_contact_me[i-1] = sum(df.contact_channel_first[inquiry_idx_i] == 'contact_me')
    df_interval.total_book_it[i-1] = sum(df.contact_channel_first[inquiry_idx_i] == 'book_it')
    df_interval.total_instant_book[i-1] = sum(df.contact_channel_first[inquiry_idx_i] == 'instant_book')

    # Percent of inquiries in each channel
    df_interval.pct_contact_me[i-1] = df_interval.total_contact_me[i-1] / df_interval.total_inquiries[i-1]
    df_interval.pct_book_it[i-1] = df_interval.total_book_it[i-1] / df_interval.total_inquiries[i-1]
    df_interval.pct_instant_book[i-1] = df_interval.total_instant_book[i-1] / df_interval.total_inquiries[i-1]

    # Percent of inquiries made by past booker or new user
    df_interval.pct_past_booker[i-1] = sum(df.guest_user_stage_first[inquiry_idx_i] == 'past_booker') / df_interval.total_inquiries[i-1]
    df_interval.pct_new_user[i-1] = sum(df.guest_user_stage_first[inquiry_idx_i] == 'new') / df_interval.total_inquiries[i-1]

    # number replied
    df_interval.total_replies[i-1] = len(reply_idx_i)

    # host reply rate
    df_interval.reply_rate[i-1] = df_interval.total_replies[i-1] / df_interval.total_inquiries[i-1]

    # average reply time
    df_interval.avg_reply_time[i-1] = np.nanmean((df.ts_reply_at_first[reply_idx_i] - df.ts_interaction_first[reply_idx_i]).dt.seconds)
    df_interval.avg_reply_time[i-1] /= (3600*24) if interval == 'day' else 3600

    # number accepted
    df_interval.total_accepted[i-1] = len(accepted_idx_i)

    # host accepted rate
    df_interval.accepted_rate[i-1] = df_interval.total_accepted[i-1] / df_interval.total_inquiries[i-1]

    # average accepted time
    df_interval.avg_accepted_time[i-1] = np.nanmean((df.ts_accepted_at_first[accepted_idx_i] - df.ts_interaction_first[accepted_idx_i]).dt.seconds)
    df_interval.avg_accepted_time[i-1] /= (3600*24) if interval == 'day' else 3600

    # number booked
    df_interval.total_booking[i-1] = len(booking_idx_i)

    # host booking rate
    df_interval.booking_rate[i-1] = df_interval.total_booking[i-1] / df_interval.total_inquiries[i-1]

    # average booking time
    df_interval.avg_booking_time[i-1] = np.nanmean((df.ts_booking_at[booking_idx_i] - df.ts_interaction_first[booking_idx_i]).dt.seconds)
    df_interval.avg_booking_time[i-1] /= (3600*24) if interval == 'day' else 3600

  return df_interval
