import pandas as pd

def generate_timeseries(df, interval='D', segments = None, values = None):

    df = df.copy()

    df['ts_interaction_first'] = pd.to_datetime(df['ts_interaction_first'])    
    df.set_index('ts_interaction_first', inplace=True)
    df.index.name = 'inquiry_date'
    
    time = df.resample(interval).count().index
    
    master_df = pd.DataFrame(index = time)
    master_df.index.name = df.index.name

    # Define a list of all the functions to be applied
    metrics_functions = [
        calculate_inquiry_rate,
        calculate_listing_rate,
        calculate_avg_user_profile_completeness,  # Special handling for this function
        calculate_host_response_rate, 
        calculate_host_approval_rate, 
        calculate_booking_conversion_rate, 
        calculate_avg_response_time, 
        calculate_avg_approval_time, 
        calculate_avg_booking_time, 
        calculate_avg_stay_time, 
        calculate_avg_engagement, 
        calculate_avg_reviews
    ]

    # Metric names without 'calculate_' prefix
    metric_names = [func.__name__.replace('calculate_', '') for func in metrics_functions]

    if segments is not None:
        for segment in segments:
            # unique_values = df[segment].unique()
            
            for value in values:
                # Initialize a DataFrame for this segment value
                # segment_df = pd.DataFrame(index=master_df.index)                
                for func, metric_name in zip(metrics_functions, metric_names):
                    if func == calculate_avg_user_profile_completeness:
                        for user_type in ['guest', 'host']:
                            # Call the function with additional user_type parameter
                            result = func(df, variable = segment, value = value, user=user_type, interval=interval)

                            master_df = master_df.join(pd.DataFrame(result, index = master_df.index), how='left')
                            
                    else:
                        # Call the function with the segment and value
                        result = func(df, variable = segment, value = value, interval=interval)
                        
                        master_df = master_df.join(pd.DataFrame(result, index = master_df.index), how='left')
                
                # # Add the segment value to the DataFrame
                # segment_df[segment] = value
                # # Concatenate this segment DataFrame to the master DataFrame
                # master_df = pd.concat([master_df, segment_df], axis=0)
            
    else:
        for func, metric_name in zip(metrics_functions, metric_names):
            if func == calculate_avg_user_profile_completeness:
                for user_type in ['guest', 'host']:
                    # Call the function with additional user_type parameter
                    result = func(df, user=user_type, interval=interval)
                    
                    master_df = master_df.join(pd.DataFrame(result, index = master_df.index), how='left')
            else:
                # Call the function for the entire dataset
                result = func(df, interval=interval)

                master_df = master_df.join(pd.DataFrame(result, index = master_df.index), how='left')
                
                
    # # Rename the index to 'inquiry_date'
    # master_df.index.name = 'inquiry_date'
    # master_df.reset_index(inplace=True)
    
    return master_df
