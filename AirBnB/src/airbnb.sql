-- We begin by creating the three tables 'contacts', 'listings', and 'users'
-- using the the respective CSV files provided.

-- Columns of 'contacts' table:
-- id_guest_anon: Anonymous ID of the guest.
-- id_host_anon: Anonymous ID of the host.
-- id_listing_anon: Anonymous ID of the listing.
-- ts_interaction_first: Timestamp of the first interaction.
-- ts_reply_at_first: Timestamp of the first reply.
-- ts_accepted_at_first: Timestamp of the first acceptance.
-- ts_booking_at: Timestamp of the booking.
-- ds_checkin_first: Timestamp of the first check-in date.
-- ds_checkout_first: Timestamp of the first check-out date.
-- m_guests: Real number representing the number of guests.
-- m_interactions: Real number representing the number of interactions.
-- m_first_message_length_in_characters: Real number representing the length of the first message in characters.
-- contact_channel_first: Text field representing the first contact channel used.
-- guest_user_stage_first: Text field representing the guest user stage.

create table 
	contacts (id_guest_anon text, 
			  id_host_anon text,
			  id_listing_anon text, 
			  ts_interaction_first timestamp, 
			  ts_reply_at_first timestamp, 
			  ts_accepted_at_first timestamp, 
			  ts_booking_at timestamp, 
			  ds_checkin_first timestamp, 
			  ds_checkout_first timestamp, 
			  m_guests real,
			  m_interactions real, 
			  m_first_message_length_in_characters real, 
			  contact_channel_first text, 
			  guest_user_stage_first text) ;

-- Columns of 'listings' table:
-- id_listing_anon: Anonymous ID of the listing.
-- room_type: Text field describing the room type.
-- listing_neighborhood: Text field describing the neighborhood of the listing.
-- total_reviews: Real number representing the total number of reviews for the listing.

create table 
	listings (id_listing_anon text, 
			  room_type text, 
			  listing_neighborhood text, 
			  total_reviews real) ;

-- Columns of 'users' table:
-- id_user_anon: Anonymous ID of the user.
-- country: Text field representing the user's country.
-- words_in_user_profile: Real number representing the number of words in the user's profile.

create table 
	users (id_user_anon text, 
  		   country text, 
 		   words_in_user_profile real) ;

-----------------------------------------------------------------------------

drop view if exists hourly ;
drop view if exists inquiries ;
drop view if exists replies ;
drop view if exists accepted ;
drop view if exists bookings ;
drop view if exists master ;

-----------------------------------------------------------------------------

-- View Name: master
-- Description: This view combines data from the 'contacts', 'listings', and 'users' tables.

create view master as
select
    c.*,                            -- All columns from the 'contacts' table
    l.room_type,                    -- The room type from the 'listings' table
    l.listing_neighborhood,          -- The neighborhood of the listing from the 'listings' table
    l.total_reviews,                -- The total number of reviews for the listing from the 'listings' table
    g.country AS guest_country,     -- The country of the guest from the 'users' table, aliased as 'guest_country'
    g.words_in_user_profile AS words_in_guest_profile,  -- Words in the guest's user profile from the 'users' table, aliased as 'words_in_guest_profile'
    h.country AS host_country,      -- The country of the host from the 'users' table, aliased as 'host_country'
    h.words_in_user_profile AS words_in_host_profile    -- Words in the host's user profile from the 'users' table, aliased as 'words_in_host_profile'
from 
    contacts as c                   -- Alias for the 'contacts' table
join 
    listings as l on c.id_listing_anon = l.id_listing_anon  -- Join 'contacts' with 'listings' using the 'id_listing_anon' column
join
    users as g on c.id_guest_anon = g.id_user_anon          -- Join 'contacts' with 'users' (guest) using the 'id_guest_anon' column
join
    users as h on c.id_host_anon = h.id_user_anon           -- Join 'contacts' with 'users' (host) using the 'id_host_anon' column
where 														-- where reply, accept, and booking dates are not greater than check-in date
	((c.ds_checkin_first is null) or (c.ds_checkin_first >= c.ts_interaction_first)) -- 
	and
	((c.ts_reply_at_first is null) or extract(epoch from c.ts_reply_at_first - c.ts_interaction_first)/(3600*24) < 0.2)	--
-- 	and
--     l.listing_neighborhood IN ('-unknown-','Copacabana','Ipanema','Leblon')
-- 	and
--     g.country IN all ('US')  
order by
	c.ts_interaction_first ;

select * from master ;

-----------------------------------------------------------------------------

create view inquiries as
	select
		date_trunc('hour', ts_interaction_first) as inquiry_time,
		count(ts_interaction_first) as total_inquiries,
		avg(m_guests) as avg_guests,
		avg(m_interactions) as avg_interactions,
		count(ts_interaction_first)::float / count(distinct id_guest_anon) as avg_inquires_per_guest,
		sum(case when room_type = 'Entire home/apt' then 1 else 0 end) as total_home_inquires,
		sum(case when room_type = 'Shared room' then 1 else 0 end) as total_shared_inquires,
		sum(case when room_type = 'Private room' then 1 else 0 end) as total_private_inquires,
		avg(m_first_message_length_in_characters) as avg_message_length,
		sum(case when contact_channel_first = 'contact_me' then 1 else 0 end) as total_contact_me,
		sum(case when contact_channel_first = 'book_it' then 1 else 0 end) as total_book_it,
		sum(case when contact_channel_first = 'instant_book' then 1 else 0 end) as total_instant_book,
		sum(case when contact_channel_first = 'contact_me' then 1 else 0 end)::float/count(contact_channel_first)::float as pct_contact_me,
		sum(case when contact_channel_first = 'book_it' then 1 else 0 end)::float/count(contact_channel_first)::float as pct_book_it_me,
		sum(case when contact_channel_first = 'instant_book' then 1 else 0 end)::float/count(contact_channel_first)::float as pct_instant_book,
		sum(case when guest_user_stage_first = 'past_booker' then 1 else 0 end)::float/count(guest_user_stage_first)::float as pct_past_booker,
		sum(case when guest_user_stage_first = 'new' then 1 else 0 end)::float/count(guest_user_stage_first)::float as pct_new_user
	from 
		master
	group by
		inquiry_time ;

select * from inquiries ;

-----------------------------------------------------------------------------

create view replies as
	select			
		date_trunc('hour', ts_reply_at_first) as reply_time,		
		count(ts_reply_at_first) as total_replies,
		avg(extract(epoch from (ts_reply_at_first - ts_interaction_first))/(3600)) as avg_reply_hours
	from 
		master
	group by
		reply_time
	order by 
		reply_time ;

select * from replies ;

-----------------------------------------------------------------------------

create view accepted as
	select
		date_trunc('hour', ts_accepted_at_first) as accepted_time,
		count(ts_accepted_at_first) as total_accepted
	from 
		master
	group by
		accepted_time
	order by
		accepted_time ;

select * from accepted ;

-----------------------------------------------------------------------------

create view bookings as
	select
		date_trunc('hour', ts_booking_at) as booking_time,
		count(ts_booking_at) as total_booking
	from 
		master
	group by
		booking_time
	order by
		booking_time ;

select * from bookings ;

-----------------------------------------------------------------------------

create view hourly as
	select		
		i.*,
		r.*,
		a.*,
		b.*,
		coalesce(i.inquiry_time, r.reply_time, a.accepted_time, b.booking_time) as time
	from inquiries as i
	join 
		replies as r on i.inquiry_time = r.reply_time
	join
		accepted as a on r.reply_time = a.accepted_time
	join
		bookings as b on a.accepted_time = b.booking_time
	order by
		i.inquiry_time ;
		
select * from hourly ;
