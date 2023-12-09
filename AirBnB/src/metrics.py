
## Inquiry Rate
def calculate_inquiry_rate(df, variable = None, value = None, interval = 'D'):

    df = df.copy()
        
    # df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])
    # df.set_index('ts_interaction_first', inplace=True)
    
    # df.index.name = 'inquiry_date'

    df_resampled_total = df.resample(interval).agg({'id_guest_anon': 'nunique'})

    if variable is not None and value is not None:
        if variable in df.columns:
            df = df[df[variable] == value]
        else:
            raise ValueError(f"Column '{variable}' not found in DataFrame")

    df_resampled = df.resample(interval).agg({'id_guest_anon': 'nunique'})

    inquiry_rate = df_resampled['id_guest_anon'] / df_resampled_total['id_guest_anon']

    inquiry_rate = inquiry_rate.fillna(0)
    
    if variable is not None and value is not None:
        inquiry_rate.name = f"inquiry_rate_{variable}_{value}"
    else:
        inquiry_rate.name = 'inquiry_rate'

    return inquiry_rate

# Listing Rate
def calculate_listing_rate(df, variable = None, value = None, interval = 'D'):
    
    df = df.copy()
    
    # df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])
    # df.set_index('ts_interaction_first', inplace=True)

    # df.index.name = 'inquiry_date'
    
    df_resampled_total = df.resample(interval).agg({'id_listing_anon': 'nunique'})

    if variable is not None and value is not None:
        if variable in df.columns:
            df = df[df[variable] == value]
        else:
            raise ValueError(f"Column '{variable}' not found in DataFrame")

    df_resampled = df.resample(interval).agg({'id_listing_anon': 'nunique'})
    
    listing_rate = (df_resampled['id_listing_anon'] / df_resampled_total['id_listing_anon'])

    listing_rate = listing_rate.fillna(0)

    if variable is not None and value is not None:
        listing_rate.name = f"listing_rate_{variable}_{value}"
    else:
        listing_rate.name = 'listing_rate'

    return listing_rate

# Average User Profile
def calculate_avg_user_profile_completeness(df, user='guest', interval='D', variable=None, value=None):
    
    df = df.copy()

    # df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])
    # df.set_index('ts_interaction_first', inplace=True)

    # df.index.name = 'inquiry_date'

    profile_length_column = f"words_in_{user}_profile"

    if profile_length_column not in df.columns:
        raise ValueError(f"Column '{profile_length_column}' not found in DataFrame")

    if variable is not None and value is not None:
        if variable in df.columns:
            df = df[df[variable] == value]
        else:
            raise ValueError(f"Column '{variable}' not found in DataFrame")
    
    avg_user_profile_length = df.resample(interval).agg({profile_length_column: 'mean'})

    avg_user_profile_length = avg_user_profile_length.fillna(0)

    avg_user_profile_length = avg_user_profile_length[profile_length_column]

    if variable is not None and value is not None:
        avg_user_profile_length.name = f"avg_{user}_profile_completeness_{variable}_{value}"
    else:
        avg_user_profile_length.name = f"avg_{user}_profile_completeness"

    return avg_user_profile_length

# Host Response Rate
def calculate_host_response_rate(df, interval='D', variable=None, value=None):
    
    df = df.copy()

    # df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])
    # df.set_index('ts_interaction_first', inplace=True)

    # df.index.name = 'inquiry_date'
    
    if variable is not None and value is not None:
        if variable in df.columns:
            df = df[df[variable] == value]
        else:
            raise ValueError(f"Column '{variable}' not found in DataFrame")

    df['inquired'] = (~df.index.isna()).astype(int)
    df['inquiry_responded'] = (~df['ts_reply_at_first'].isna()).astype(int) * df['inquired']

    required_columns = ['inquired', 'inquiry_responded'] 
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    df_resampled = df.resample(interval).agg({'inquired': 'sum', 'inquiry_responded': 'sum'})
    
    host_response_rate = df_resampled['inquiry_responded'] / df_resampled['inquired']
    
    host_response_rate = host_response_rate.fillna(0)

    if variable is not None and value is not None:
        host_response_rate.name = f"host_response_rate_{variable}_{value}"
    else:
        host_response_rate.name = 'host_response_rate'

    return host_response_rate

# Host Approval Rate
def calculate_host_approval_rate(df, interval='D', variable=None, value=None):
    
    df = df.copy()

    # df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])
    # df.set_index('ts_interaction_first', inplace=True)

    if variable is not None and value is not None:
        if variable in df.columns:
            df = df[df[variable] == value]
        else:
            raise ValueError(f"Column '{variable}' not found in DataFrame")

    df = df[df['contact_channel_first'] != 'instant_book']
    
    df['inquired'] = (~df.index.isna()).astype(int)
    df['inquiry_accepted'] = (~df['ts_accepted_at_first'].isna()).astype(int) * df['inquired']

    required_columns = ['inquired', 'inquiry_accepted'] 
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    df_resampled = df.resample(interval).agg({'inquired': 'sum', 'inquiry_accepted': 'sum'})

    host_approval_rate = df_resampled['inquiry_accepted'] / df_resampled['inquired']

    host_approval_rate = host_approval_rate.fillna(0)

    if variable is not None and value is not None:
        host_approval_rate.name = f"host_approval_rate_{variable}_{value}"
    else:
        host_approval_rate.name = 'host_approval_rate'

    return host_approval_rate

# Booking Conversion Rate
def calculate_booking_conversion_rate(df, interval='D', variable=None, value=None):
    
    df = df.copy()

    # df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])
    # df.set_index('ts_interaction_first', inplace=True)

    # df.index.name = 'inquiry_date'
    
    if variable is not None and value is not None:
        if variable in df.columns:
            df = df[df[variable] == value]
        else:
            raise ValueError(f"Column '{variable}' not found in DataFrame")

    df['inquired'] = (~df.index.isna()).astype(int)
    df['inquiry_booked'] = (~df['ts_booking_at'].isna()).astype(int) * df['inquired']

    required_columns = ['inquired', 'inquiry_booked'] 
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    df_resampled = df.resample(interval).agg({'inquired': 'sum', 'inquiry_booked': 'sum'})

    booking_conversion_rate = df_resampled['inquiry_booked'] / df_resampled['inquired']

    booking_conversion_rate = booking_conversion_rate.fillna(0)

    if variable is not None and value is not None:
        booking_conversion_rate.name = f"booking_conversion_rate_{variable}_{value}"
    else:
        booking_conversion_rate.name = 'booking_conversion_rate'

    return booking_conversion_rate

# Average Response Time
def calculate_avg_response_time(df, interval='D', variable=None, value=None):
    df = df.copy()

    # Convert the time columns to datetime
    # df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])
    df['ts_reply_at_first'] = pd.to_datetime(df['ts_reply_at_first'])
    # df.set_index('ts_interaction_first', inplace=True)

    # df.index.name = 'inquiry_date'
    
    # Apply filtering if a variable and value are provided
    if variable is not None and value is not None:
        if variable in df.columns:
            df = df[df[variable] == value]
        else:
            raise ValueError(f"Column '{variable}' not found in DataFrame")

    # Calculate the response time in hours
    df['response_time'] = (df['ts_reply_at_first'] - df.index).dt.total_seconds() / 3600

    # Resample and calculate the average response time in each interval
    avg_response_time = df.resample(interval).agg({'response_time': 'mean'})

    # Handle potential NaN values
    avg_response_time = avg_response_time.fillna(0)

    avg_response_time = avg_response_time['response_time']
    
    if variable is not None and value is not None:
        avg_response_time.name = f"avg_response_time_{variable}_{value}"
    else:
        avg_response_time.name = 'avg_response_time'

    return avg_response_time

# Average Approval Time
def calculate_avg_approval_time(df, interval='D', variable=None, value=None):
    df = df.copy()

    # Convert the time columns to datetime
    # df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])
    df['ts_accepted_at_first'] = pd.to_datetime(df['ts_accepted_at_first'])
    # df.set_index('ts_interaction_first', inplace=True)

    # Apply filtering if a variable and value are provided
    if variable is not None and value is not None:
        if variable in df.columns:
            df = df[df[variable] == value]
        else:
            raise ValueError(f"Column '{variable}' not found in DataFrame")

    # Calculate the approval time in hours
    df['approval_time'] = (df['ts_accepted_at_first'] - df.index).dt.total_seconds() / 3600

    # Resample and calculate the average approval time in each interval
    avg_approval_time = df.resample(interval).agg({'approval_time': 'mean'})

    # Handle potential NaN values
    avg_approval_time = avg_approval_time.fillna(0)

    avg_approval_time = avg_approval_time['approval_time']

    if variable is not None and value is not None:
        avg_approval_time.name = f"avg_approval_time_{variable}_{value}"
    else:
        avg_approval_time.name = 'avg_approval_time'

    return avg_approval_time

# Average Booking Time
def calculate_avg_booking_time(df, interval='D', variable=None, value=None):
    df = df.copy()

    # Convert the time columns to datetime
    # df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])
    df['ts_booking_at'] = pd.to_datetime(df['ts_booking_at'])
    # df.set_index('ts_interaction_first', inplace=True)

    # df.index.name = 'inquiry_date'
    
    # Apply filtering if a variable and value are provided
    if variable is not None and value is not None:
        if variable in df.columns:
            df = df[df[variable] == value]
        else:
            raise ValueError(f"Column '{variable}' not found in DataFrame")
    
    # Calculate the approval time in hours
    df['booking_time'] = (df['ts_booking_at'] - df.index).dt.total_seconds() / 3600

    # Resample and calculate the average booking time in each interval
    avg_booking_time = df.resample(interval).agg({'booking_time': 'mean'})

    # Handle potential NaN values
    avg_booking_time = avg_booking_time.fillna(0)

    avg_booking_time = avg_booking_time['booking_time']

    if variable is not None and value is not None:
        avg_booking_time.name = f"avg_booking_time_{variable}_{value}"
    else:
        avg_booking_time.name = 'avg_booking_time'

    return avg_booking_time

# Average Stay Time
def calculate_avg_stay_time(df, interval='D', variable=None, value=None):
    
    df = df.copy()

    # df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])
    
    df['ds_checkin_first'] = pd.to_datetime(df['ds_checkin_first'])
    df['ds_checkout_first'] = pd.to_datetime(df['ds_checkout_first'])

    # df.set_index('ts_interaction_first', inplace=True)

    if variable is not None and value is not None:
        if variable in df.columns:
            df = df[df[variable] == value]
        else:
            raise ValueError(f"Column '{variable}' not found in DataFrame")

    df['stay_time'] = (df['ds_checkout_first'] - df['ds_checkin_first']).dt.total_seconds() / 3600 / 24

    avg_stay_time = df.resample(interval).agg({'stay_time': 'mean'})

    avg_stay_time = avg_stay_time.fillna(0)

    avg_stay_time = avg_stay_time['stay_time']

    if variable is not None and value is not None:
        avg_stay_time.name = f"avg_stay_time_{variable}_{value}"
    else:
        avg_stay_time.name = 'avg_stay_time'

    return avg_stay_time

# Average Engagement
def calculate_avg_engagement(df, interval='D', variable=None, value=None):
    
    df = df.copy()
    
    # df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])
    # df.set_index('ts_interaction_first', inplace=True)
    
    # df.index.name = 'inquiry_date'
    
    if variable is not None and value is not None:
        if variable in df.columns:
            df = df[df[variable] == value]
        else:
            raise ValueError(f"Column '{variable}' not found in DataFrame")

    avg_engagement = df.resample(interval).agg({'m_interactions': 'mean'})

    avg_engagement = avg_engagement.fillna(0)

    avg_engagement = avg_engagement['m_interactions']

    if variable is not None and value is not None:
        avg_engagement.name = f"avg_engagement_{variable}_{value}"
    else:
        avg_engagement.name = 'avg_engagement'

    return avg_engagement

# IQR Outliers
def iqr_outlier(df_col, ordered=False):
    
    df_col = df_col.copy()
    
    median = round(df_col.median()) if df_col.dtype == int else df_col.median()
    iqr = df_col.quantile(0.75) - df_col.quantile(0.25)
    lower_threshold = df_col.quantile(0.25) - 1.5 * iqr
    upper_threshold = df_col.quantile(0.75) + 1.5 * iqr
    
    if not ordered:
        outlier_condition = ~df_col.between(lower_threshold, upper_threshold)
        df_col.loc[outlier_condition] = median
    else:
        # Find the indices of the outliers
        outlier_indices = df_col.loc[~df_col.between(lower_threshold, upper_threshold), :]
        
        # Replace outliers by taking the mean of the neighbors
        for index in outlier_indices:
            idx = df_col.index.get_loc(index)
            if idx > 0 and idx < (len(df_col) - 1):
                df_col.at[index] = (df_col.iloc[idx - 1] + df_col.iloc[idx + 1]) / 2

    return df_col
