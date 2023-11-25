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

  df_interval = pd.DataFrame({'total_inquiries': zeros_,
                              'total_replies': zeros_,
                              'total_accepted': zeros_,
                              'total_booking': zeros_,
                              'contact_me_rate': zeros_,
                              'book_it_rate': zeros_,
                              'instant_book_rate': zeros_,
                              'guest_return_rate': zeros_,
                              'guest_message_rate': zeros_,
                              'words_in_guest_profile': zeros_,
                              'reply_rate': zeros_,
                              'reply_time': nans_,
                              'accepted_rate': zeros_,
                              'accepted_time': nans_,
                              'reply_conversion_rate': zeros_,
                              'accepted_conversion_rate': zeros_,
                              'booking_rate': zeros_,
                              'booking_time': nans_,
                              'listing_rate': zeros_,
                              'interaction': zeros_},
                              index = time[1:])

  inquiry_idx_I = []


  for i in range(1, len(time)):
    min_time_i, max_time_i = time[i-1], time[i]

    interval_len = sum((df.ts_interaction_first > min_time_i) & (df.ts_interaction_first <= max_time_i))

    inquiry_idx_i = np.where(~df.ts_interaction_first.isnull() & (df.ts_interaction_first > min_time_i) & (df.ts_interaction_first <= max_time_i))[0]
    reply_idx_i = np.where(~df.ts_reply_at_first.isnull() & (df.ts_reply_at_first > min_time_i) & (df.ts_reply_at_first <= max_time_i))[0]
    accepted_idx_i = np.where(~df.ts_accepted_at_first.isnull() & (df.ts_accepted_at_first > min_time_i) & (df.ts_accepted_at_first <= max_time_i))[0]
    booking_idx_i = np.where(~df.ts_booking_at.isnull() & (df.ts_booking_at > min_time_i) & (df.ts_booking_at <= max_time_i))[0]

    total_contact_me_i = sum(df.contact_channel_first[inquiry_idx_i] == 'contact_me')
    total_book_it_i = sum(df.contact_channel_first[inquiry_idx_i] == 'book_it')
    total_instant_book_i = sum(df.contact_channel_first[inquiry_idx_i] == 'instant_book')

    total_inquiries_i = len(inquiry_idx_i)
    total_replies_i = len(reply_idx_i)
    total_accepted_i = len(accepted_idx_i)
    total_booking_i = len(booking_idx_i)

    inquiry_idx_I.append(inquiry_idx_i)

    # total inquiries
    df_interval.loc[df_interval.index[i-1], 'total_inquiries'] = total_inquiries_i
    # total replies
    df_interval.loc[df_interval.index[i-1], 'total_replies'] = total_replies_i
    # total accepted
    df_interval.loc[df_interval.index[i-1], 'total_accepted'] = total_accepted_i
    # total booking
    df_interval.loc[df_interval.index[i-1], 'total_booking'] = total_booking_i

    # 'contact me' rate
    df_interval.loc[df_interval.index[i-1], 'contact_me_rate'] = total_contact_me_i / total_inquiries_i
    # 'book it' rate
    df_interval.loc[df_interval.index[i-1], 'book_it_rate'] = total_book_it_i / total_inquiries_i
    # 'instant book' rate
    df_interval.loc[df_interval.index[i-1], 'instant_book_rate'] = total_instant_book_i / total_inquiries_i

    # guest return rate
    past_booker_idx_i = inquiry_idx_i[np.where(df.guest_user_stage_first[inquiry_idx_i] == 'past_booker')[0]]
    df_interval.loc[df_interval.index[i-1], 'guest_return_rate'] = df.id_guest_anon[past_booker_idx_i].unique().size / df.id_guest_anon[inquiry_idx_i].unique().size

    # guest message rate
    df_interval.loc[df_interval.index[i-1], 'guest_message_rate'] = df.m_first_message_length_in_characters[inquiry_idx_i].sum() / total_inquiries_i

    # guest profile length
    df_interval.loc[df_interval.index[i-1], 'words_in_guest_profile'] = df.words_in_guest_profile[inquiry_idx_i].mean()

    # reply rate
    df_interval.loc[df_interval.index[i-1], 'reply_rate']  = total_replies_i / total_inquiries_i
    # reply time
    df_interval.loc[df_interval.index[i-1], 'reply_time'] = (df.ts_reply_at_first[inquiry_idx_i] - df.ts_interaction_first[inquiry_idx_i]).dropna().mean().seconds
    df_interval.loc[df_interval.index[i-1], 'reply_time'] /= 3600 if interval == 'hour' else 3600*24

    # accepted rate
    df_interval.loc[df_interval.index[i-1], 'accepted_rate']  = total_accepted_i / total_inquiries_i
    # accepted time
    df_interval.loc[df_interval.index[i-1], 'accepted_time'] = (df.ts_accepted_at_first[inquiry_idx_i] - df.ts_interaction_first[inquiry_idx_i]).dropna().mean().seconds
    df_interval.loc[df_interval.index[i-1], 'accepted_time'] /= 3600 if interval == 'hour' else 3600*24

    # booking rate
    df_interval.loc[df_interval.index[i-1], 'booking_rate']  = total_booking_i / total_inquiries_i
    # booking time
    df_interval.loc[df_interval.index[i-1], 'booking_time'] = (df.ts_booking_at[inquiry_idx_i] - df.ts_interaction_first[inquiry_idx_i]).dropna().mean().seconds
    df_interval.loc[df_interval.index[i-1], 'booking_time'] /= 3600 if interval == 'hour' else 3600*24

    # reply conversion rate
    df_interval.loc[df_interval.index[i-1], 'reply_conversion_rate'] = total_booking_i / total_replies_i

    # accepted conversion rate
    df_interval.loc[df_interval.index[i-1], 'accepted_conversion_rate'] = total_booking_i / total_accepted_i

    # listing rate
    df_interval.loc[df_interval.index[i-1], 'listing_rate'] = len(df.id_listing_anon[inquiry_idx_i].unique())/len(df.id_guest_anon[inquiry_idx_i].unique())

    # interaction
    df_interval.loc[df_interval.index[i-1], 'interaction'] = df.m_interactions[inquiry_idx_i].sum()/len(set(df.id_guest_anon[inquiry_idx_i]) | set(df.id_host_anon[inquiry_idx_i]))

  return df_interval
