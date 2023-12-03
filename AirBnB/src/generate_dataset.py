import pandas as pd

def generate_dataset(df: pd.DataFrame,
                     interval: str = 'D',
                     room_types: list = None,
                     guest_countries: list = None,
                     listing_neighborhoods: list = None, 
                     fix_outliers = False):
    
    df_ts = df.copy()
    
    if room_types is not None:
        df_ts = df_ts[df_ts['room_type'].isin(room_types)]
    
    if guest_countries is not None:
        df_ts = df_ts[df_ts['guest_country'].isin(guest_countries)]
    
    if listing_neighborhoods is not None:
        df_ts = df_ts[df_ts['listing_neighborhood'].isin(listing_neighborhoods)]
    
    df_ts_non_instant = df_ts[df_ts['contact_channel_first'] != 'instant_book'].copy()

    # Whether a booking occured or not
    df_ts['booked'] = (~df_ts['ts_booking_at'].isna()).astype(int)

    # Whether a guest is a previous booker or not
    df_ts['past_booker'] = (df_ts['guest_user_stage_first'] == 'past_booker').astype(int)
    df_ts['past_booker_booked'] = df_ts['past_booker'] * df_ts['booked']

    # Whether a non-instant booking occured or not
    df_ts_non_instant['booked'] = (~df_ts_non_instant['ts_booking_at'].isna()).astype(int)    

    # Whether a non-instant guest is a previous booker or not
    df_ts_non_instant['past_booker'] = (df_ts_non_instant['guest_user_stage_first'] == 'past_booker').astype(int)
    df_ts_non_instant['past_booker_booked'] = df_ts_non_instant['past_booker'] * df_ts_non_instant['booked']

    # Time for the host to respond to non-instant inquiry
    df_ts_non_instant['response_time'] = df_ts_non_instant['ts_reply_at_first'] - df_ts_non_instant['ts_interaction_first']
    
    df_ts.set_index('ts_interaction_first', inplace = True)
    df_ts_non_instant.set_index('ts_interaction_first', inplace = True)
    
    interval_data = df_ts.resample(interval).agg({
        'id_guest_anon': 'count', # total inquiries
        'booked': 'sum', # total booked
        'past_booker': 'sum', # total previous bookers
        'past_booker_booked': 'sum' # total bookings by previous bookers
        })
    
    interval_data_non_instant = df_ts_non_instant.resample(interval).agg({
        'id_guest_anon': 'count', # total non-instant inquiries
        'ts_reply_at_first': 'count',
        'booked': 'sum', # total non-instant booked
        'past_booker': 'sum', # total non-instant previous nbookers
        'past_booker_booked': 'sum', # total non-instant bookings by previous bookers
        'm_first_message_length_in_characters': 'sum', # total guest message length
        'm_interactions': 'sum', # total guest-host interaction
        'response_time': 'mean' # average response time
        })
    
    interval_data_final = pd.DataFrame()
    interval_data_final.index = interval_data.index

    ## Metrics
    # Conversion Rate 
    interval_data_final['cr'] = interval_data['booked'] / interval_data['id_guest_anon']

    # Average Response Time
    interval_data_final['rt'] = interval_data_non_instant['response_time'].dt.total_seconds()
    interval_data_final['rt'] /= 3600 if interval == 'D' else 60

    # Average Response Rate
    interval_data_final['rr'] = interval_data_non_instant['ts_reply_at_first'] / interval_data_non_instant['id_guest_anon']

    # Repeat Gueast Rate
    interval_data_final['rgr'] = interval_data_non_instant['past_booker'] / interval_data_non_instant['id_guest_anon']

    # Repeat booking Rate
    interval_data_final['rbr'] = interval_data_non_instant['past_booker_booked'] / interval_data_non_instant['booked']

    # Average length of guests' first message
    interval_data_final['msg'] = interval_data_non_instant['m_first_message_length_in_characters'] / interval_data_non_instant['id_guest_anon']

    # Average length of guest-host interaction
    interval_data_final['ghi'] = interval_data_non_instant['m_interactions'] / interval_data_non_instant['id_guest_anon']

    interval_data_final['day'] = interval_data_final.index.strftime('%A')

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    interval_data_final['day'] = pd.Categorical(interval_data_final['day'], categories = day_order, ordered = True)

    if fix_outliers:
        numeric_columns = interval_data_final.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            median_i = interval_data_final[col].median()
            iqr = interval_data_final[col].quantile(0.75) - interval_data_final[col].quantile(0.25)
            lower_threshold = interval_data_final[col].quantile(0.25) - 1.5 * iqr
            upper_threshold = interval_data_final[col].quantile(0.75) + 1.5 * iqr
            outlier_condition = ~interval_data_final[col].between(lower_threshold, upper_threshold)
            interval_data_final.loc[outlier_condition, col] = median_i

    return interval_data_final
