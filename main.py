# Main datasets for Airbnb in Ireland sourced here: http://insideairbnb.com/get-the-data.html
# Inside Airbnb scraped these datasets on 23/10/2021 from the Airbnb website.
# There are 3 datasets listings, reviews, and calendar.
# Note that data was scraped during the Covid pandemic amidst high uncertainty about hospitality/tourism/travel.

# First look at reviews dataset as best indicator of how busy Airbnb properties in Ireland are.
# Assumption made by Inside Airbnb and many sources on the web is that 50% of guests leave reviews
# This is the best metric available to indicate likely past occupancy.

# Import initial packages needed
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Import Reviews dataset
df_reviews = pd.read_csv(r"C:\Users\jenni\Documents\Datasets\Ireland Oct 21 Airbnb\reviews.csv")

# Check columns and data types
df_reviews.info()

# Take a look at the data
print(df_reviews.head())

# Rename columns to avoid confusion
df_reviews.columns = ['listing_id', 'review_id', 'review_date', 'reviewer_id', 'reviewer_name', 'reviewer_comments']

# Change 'date' data type to datetime, create new columns with the year and the calendar quarter
df_reviews['review_date'] = pd.to_datetime(df_reviews['review_date'])
df_reviews['year'] = pd.to_datetime(df_reviews['review_date']).dt.to_period('Y')
df_reviews['quarter_new'] = df_reviews['review_date'].dt.to_period('Q')

# Drop reviews pre 2016 and also drop reviews in early Q4 2021 as the data for Q4 2021 is only partial.
df_reviews.drop(df_reviews[df_reviews['review_date'] < '2016-01-01'].index, inplace=True)
df_reviews.drop(df_reviews[df_reviews['review_date'] > '2021-09-30'].index, inplace=True)

# Use a groupby to get the count of reviews by quarter and the number of unique property listings this relates to
grp_dfreviews = df_reviews.groupby('quarter_new').agg({'listing_id': ['nunique'], 'quarter_new': ['count']})

# flatten every level of MultiIndex
grp_dfreviews.reset_index(inplace=True)

# Rename columns for clarity
grp_dfreviews.columns = ['Year_Qtr', 'Listings_Count', 'Reviews_Count']

# Review
print(grp_dfreviews.head())

# Convert the quarter_new field back to a string to plot the data
grp_dfreviews['Year_Qtr'] = grp_dfreviews['Year_Qtr'].astype('str')

# Plot a line graph to show the trend in reviews and by implication stays at Airbnb properties in Ireland from 2016
# Also plot number of unique listings to which the reviews relate

# Set x, y1 and y2
x = grp_dfreviews['Year_Qtr']
y1 = grp_dfreviews['Reviews_Count']
y2 = grp_dfreviews['Listings_Count']

# Plot Graph
plt.figure(figsize=(16, 5))
plt.plot(x, y1, marker="o", linestyle="-", color="tab:orange", label="reviews")
plt.plot(x, y2, marker="*", linestyle="-.", color="b", label="unique listings with reviews")
plt.xticks(rotation=45)
plt.title("Reviews over Time")
plt.xlabel("Quarter")
plt.ylabel("Reviews/Unique Listings with Reviews")
plt.legend()
plt.show()

# What regions in Ireland are busiest in terms of stays at Airbnb listed properties?
# In what regions are Aibnb properties busiest?
# How does the regional spread of Airbnb stays compare to the regional spread of tourist bednights in Ireland?
# Look at the locations of 2019 review data to get a view of this in the most recent full year pre pandemic.

# Filter on df_reviews to get a sub dataset with reviews in 2019 only
df_reviews_2019 = df_reviews[df_reviews['year'] == '2019'].copy()

# Merge df_reviews_2019 with the listings dataset to bring in location fields.

# Import the listings dataset
df_listings = pd.read_csv(r"C:\Users\jenni\Documents\Datasets\Ireland Oct 21 Airbnb\listings.csv")

# Take a look at the data
print(df_listings.head())

# Take a look at the number of region options under region_parent_name.
df_listings['region_parent_name'].value_counts()

# This is 1 for each county and >1 for counties Dublin, Galway and Cork.
# Long list but keep it for now as there is a lot of variation in the tourist offering across cities and counties.

# Remove the string ' County Council' and ' Council' from the region_parent_name so this doesn't clutter graphs etc.
df_listings['region_parent_name'] = df_listings['region_parent_name'].str.replace(' County Council', '')
df_listings['region_parent_name'] = df_listings['region_parent_name'].str.replace(' Council', '')

# Take a look at the split under region_parent_parent_name.
# This is used by Failte Ireland so will be a point of comparison to Failte Ireland data.
df_listings['region_parent_parent_name'].value_counts()

# Merge df_reviews_2019 with df_listings to bring region fields in to reviews data
df_reviews_2019 = df_reviews_2019.merge(df_listings[['id', 'region_parent_name', 'region_parent_parent_name']],
                                        how='inner', left_on='listing_id', right_on='id').drop(columns=['id'])

# Use a groupby to get thecount of reviews by region_parent_name in 2019 and the count of related unique listings
grp_dfreviews_2019 = df_reviews_2019.groupby(['region_parent_parent_name', 'region_parent_name']).\
        agg({'listing_id': ['nunique'], 'review_id': ['count']}).copy()

# flatten every level of MultiIndex
grp_dfreviews_2019.reset_index(inplace=True)

# Rename columns for clarity
grp_dfreviews_2019.columns = ['NUTS3_Region', 'region_parent_name', 'Listings_Count', 'Reviews_Count']
print(grp_dfreviews_2019.head())

# Sort result by County Council review count in desc order
grp_dfreviews_2019.sort_values(by=['Reviews_Count'], ascending=False, inplace=True)

# Rename region_parent_parent_name in df_listings for consistency
df_listings.rename(columns={'region_parent_parent_name': 'NUTS3_Region'}, inplace=True)

# Add in an avg number of reviews per listing by region_parent_name
grp_dfreviews_2019['avg_reviews_perlisting'] = \
    round(grp_dfreviews_2019['Reviews_Count']/grp_dfreviews_2019['Listings_Count'], 1)

# Check
print(grp_dfreviews_2019.head())
# How many reviews in total in 2019?
print(grp_dfreviews_2019['Reviews_Count'].sum())

# Plot horizontal bar chart showing the number of reviews by City or County Council Region
sns.set_style('darkgrid')
plt.figure(figsize=(16, 5))
p = sns.barplot(x='Reviews_Count', y='region_parent_name', data=grp_dfreviews_2019)
p.set_xlabel("Number of Reviews in 2019", fontsize=12)
p.set_ylabel("City or County Council/Region", fontsize=12)
plt.title("Number of Reviews in 2019 by County Council or City Council Region")
plt.show()

# Plot Horizontal Bar Chart showing the average number of reviews per listing by City or County Council Region
grp_dfreviews_2019.sort_values(by=['avg_reviews_perlisting'], ascending=False, inplace=True)
plt.figure(figsize=(16, 5))
p = sns.barplot(x='avg_reviews_perlisting', y='region_parent_name', data=grp_dfreviews_2019)
p.set_xlabel("Average Number of Reviews per Listing in 2019", fontsize=12)
p.set_ylabel("City or County Council/Region", fontsize=12)
plt.title("Average Number of Reviews Per Listing with Reviews in 2019 by County Council or City Council Region")
plt.show()

# To compare the Airbnb reviews/stays by region to the overall tourist bednights by region, I have aggregated stats
# from Failte Ireland, Tourism Ireland and the CSO.  The regions used are 'NUTS 3' which is in the review data now,
# except Mid-East and Midlands are combined in the Failte Ireland data.

# Change values Mid-East and Midlands to Mid-East_Midlands
grp_dfreviews_2019['NUTS3_Region'] = grp_dfreviews_2019['NUTS3_Region'].str.replace('Mid-East', 'Midlands')
grp_dfreviews_2019['NUTS3_Region'] = grp_dfreviews_2019['NUTS3_Region'].str.replace('Midlands', 'Mid-East_Midlands')

# Use a groupby to get the count of reviews by NUTS3_Region
grp_dfreviews_2019.drop(columns=(['region_parent_name', 'Listings_Count', 'avg_reviews_perlisting']), axis=1,
                        inplace=True)
grp_NUTS3 = grp_dfreviews_2019.groupby('NUTS3_Region').sum().copy()

# to flatten the multi-level columns
grp_NUTS3.reset_index(inplace=True)
print(grp_NUTS3)

# Import estimated bednights 2019 by region from Failte Ireland, Tourism Ireland and the CSO.
# These are in thousands of nights.
df_bednights = \
    pd.read_csv(r"C:\Users\jenni\Documents\Datasets\Ireland Oct 21 Airbnb\Failte Ireland\Tourist_Bednights.csv")

# Merge df_bednights with the Airbnb review data by region for 2019
df_bednights_reviews = df_bednights.merge(grp_NUTS3[['Reviews_Count', 'NUTS3_Region']], how='inner', on='NUTS3_Region')

# Plot a graph with two Y axis showing the 2019 bednights by region on a bar chart on one axis
# and the Airbnb reviews by region on a line graph on the other Y axis.

# Set x1, y1, y2
x1 = df_bednights_reviews['NUTS3_Region']
y1 = df_bednights_reviews['Tourist_Bednights_2019_Approx']
y2 = df_bednights_reviews['Reviews_Count']

# Plot graph
plt.figure(figsize=(15, 8))
ax = sns.barplot(x=x1, y=y1, data=df_bednights_reviews)
ax2 = ax.twinx()
sns.lineplot(x=x1, y=y2, data=df_bednights_reviews, marker='o', color='crimson', lw=3, ax=ax2)
plt.show()

# Take a look at the seasonality in 2019 by region_parent_name to see if seasonality varies across regions.
# Use 2019 review data as full year pre pandemic.

# Use a groupby to group the data to see the number of reviews by region by quarter
grp_dfreviews19_Qtr = df_reviews_2019.groupby(['region_parent_name', 'quarter_new'])[['review_id']].count()

# to flatten the multi-level columns
grp_dfreviews19_Qtr.reset_index(inplace=True)

print(grp_dfreviews19_Qtr.head())

# Change quarter_new from date to string to graph
grp_dfreviews19_Qtr['quarter_new'] = grp_dfreviews19_Qtr['quarter_new'].astype('str')
# Rename columns
grp_dfreviews19_Qtr.columns = ['region_parent_name', 'quarter', 'review_count']

# Plot the number of reviews by region across the 4 quarters of 2019
sea = sns.FacetGrid(grp_dfreviews19_Qtr, col='region_parent_name', col_wrap=6, sharey=False)
sea.map(sns.lineplot, 'quarter', 'review_count')

# The listing dataset shows a price for each listing, the base price set by the host as seen on the
# profile page on Airbnb as it was on 23rd October 2021.

# Review columns and data types in df_listings
df_listings.info()

# Drop any columns which are empty, this should lose 4 columns
df_listings.dropna(how='all', axis=1, inplace=True)

# Drop other columns deemed immediately not relevant to planned project
columns_to_drop = ['listing_url', 'scrape_id', 'last_searched', 'last_scraped', 'name', 'neighborhood_overview',
                   'picture_url', 'host_id', 'host_url', 'host_name', 'host_location', 'host_thumbnail_url',
                   'host_picture_url', 'host_neighbourhood', 'host_listings_count', 'host_total_listings_count',
                   'host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood',
                   'maximum_minimum_nights', 'minimum_maximum_nights', 'minimum_minimum_nights',
                   'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'has_availability',
                   'availability_30', 'availability_60', 'availability_90', 'calendar_last_scraped',
                   'number_of_reviews_l30d', 'calculated_host_listings_count',
                   'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
                   'calculated_host_listings_count_shared_rooms', 'region_id', 'region_parent_id',
                   'region_parent_parent_id', 'reviews_per_month']
df_listings.drop(columns=columns_to_drop, inplace=True)

# Check count remaining columns
print(df_listings.shape)

# The price field is not an integer or float, but an object dtype, review.
print(df_listings['price'][0:5])

# Convert to string and remove $ symbol
df_listings['price'] = df_listings['price'].astype(str)
df_listings['price'] = df_listings['price'].str.replace('$', '', regex=True)

# Convert back to float.  This initially gives an error as there is a comma in one row, remove comma first.
df_listings['price'] = df_listings['price'].str.replace(',', '', regex=True)
df_listings['price'] = df_listings['price'].astype(float)

# There are no null values in the number_of_reviews field but only 22,538 non-null values in the first and last
# review fields. 25,977 rows - 22,538 = 3,439

# Check if this is due to listings with zero reviews
print(df_listings['number_of_reviews'][df_listings['number_of_reviews'] == 0].count())

# Drop Listings with zero reviews - property has not been booked and data is not useful for insights
df_listings.drop(df_listings[df_listings['number_of_reviews'] == 0].index, inplace=True)

# Also drop listings where last review was pre 2019, the last normal year pre pandemic
df_listings.drop(df_listings[df_listings['last_review'] <= '2019-01-01'].index, inplace=True)

# Convert columns to date time
df_listings['first_review'] = pd.to_datetime(df_listings['first_review'])
df_listings['last_review'] = pd.to_datetime(df_listings['last_review'])
df_listings['host_since'] = pd.to_datetime(df_listings['host_since'])

# Convert values in host_about to either 1 or 0 i.e. they have completed this field or
# not as this may add a personal touch that may make a difference
df_listings.host_about.fillna(0, inplace=True)
df_listings['host_about'] = df_listings['host_about'].apply(lambda z: 1 if z != 0 else 0)

# Check values in instant_bookable field
print(df_listings['instant_bookable'].value_counts())

# Convert to numerical column, f = False = 0, t = True = 1
d = {'f': 0, 't': 1}
df_listings['instant_bookable'] = df_listings['instant_bookable'].map(d).fillna(df_listings['instant_bookable'])

# Initial boxplot showed that the price for almost all properties was under €2,500 per night.
# There were a few listings with price over €20,000/night, and 4 prices of €8,000, €4,000, €3,500 & €2,739/night.

# Review listings with price > 2500 per night
print(df_listings.loc[df_listings['price'] > 2500, 'price'].count())
print(df_listings.loc[df_listings['price'] > 2500])

# Drop listings with price over 2,500 per night as these are errors plus two properties accommodating 16 people
df_listings.drop(df_listings[df_listings['price'] > 2500].index, inplace=True)

# Create boxplot to show overall distribution of nightly price for listings active since the start of 2019
plt.figure(figsize=(15, 15))
sns.color_palette("flare", as_cmap=True)
sns.boxplot(x=df_listings['room_type'], y=df_listings['price'])
plt.yticks(range(0, 2600, 100))
plt.show()

# Check counts of each room_type before proceeding
print(df_listings['room_type'].value_counts())

# Check how many entire homes have a price > €300 per night aside from the two homes accommodating 16 removed
print(df_listings.loc[(df_listings['room_type'] == 'Entire home/apt') & (df_listings['price'] > 300), 'id'].count())
# Check how many people on average these listings say they accommodate
print(df_listings.loc[(df_listings['room_type'] == 'Entire home/apt') &
                      (df_listings['price'] > 300), 'accommodates'].mean())

# Only 819 (+2) of the entire homes which have been active in or after 2019 have a price > €300 per night.
# On average these claim to accommodate 9.7 people.
# The majority of the homes are priced in a relatively narrow range.

# Create a copy of df_listings with prices up to 1000 to get a clearer boxplot
df_temp = df_listings.loc[df_listings['price'] <= 1000].copy()

# Create boxplot to show overall distribution of nightly price for listings active since the start of 2019
# by NUTS3 Region
plt.figure(figsize=(15, 15))
sns.color_palette("flare", as_cmap=True)
sns.boxplot(x=df_temp['NUTS3_Region'], y=df_temp['price'], hue=df_temp['room_type'])
plt.yticks(range(0, 1100, 100))
plt.show()

# Focussing on entire home/apts, we can see that the Dublin region has the highest prices.
# The Midlands and Border regions have the lowest prices.
# The South-West, West and Mid-East have similar median prices and have similar ranges for the top 25% of prices.

# Have a look at Seasonality & Variability of Pricing by Day of Week & By Region
# Import dataset
df_calendar = pd.read_csv(r"C:\Users\jenni\Documents\Datasets\Ireland Oct 21 Airbnb\calendar.csv")

# Take a look
print(df_calendar.head())

# Check data types
df_calendar.info()

# Drop columns not needed for analysis
df_calendar.drop(columns=['available', 'adjusted_price', 'minimum_nights', 'maximum_nights'], inplace=True)

# Convert price field to string
df_calendar['price'] = df_calendar['price'].astype('string')
# Remove dollar sign and commas in case of any
df_calendar['price'] = df_calendar['price'].str.replace('$', '', regex=True)
df_calendar['price'] = df_calendar['price'].str.replace(',', '', regex=True)
# Convert price column to float dtype
df_calendar['price'] = df_calendar['price'].astype(float)

# Change type of date field to date
df_calendar['date'] = pd.to_datetime(df_calendar['date'])
# Add quarter column to df_calendar
df_calendar['quarter'] = df_calendar['date'].dt.to_period('Q')
# Change quarter column to string dtype
df_calendar['quarter'] = df_calendar['quarter'].astype('str')
# Remove the year from the quarter column so this way partial data for Q4 2021 and 2022 will combine to
# make a full Q4
df_calendar['quarter'] = df_calendar['quarter'].str.replace('2021', '')
df_calendar['quarter'] = df_calendar['quarter'].str.replace('2022', '')

df_calendar.info()

# Look at the seasonality of pricing
qtrly_avg = df_calendar.groupby(df_calendar['quarter'], sort=False)[['price']].mean()
qtrly_avg.sort_values(by='quarter', axis=0, ascending=True, inplace=True)
qtrly_avg.plot(kind='barh', figsize=(12, 7))
plt.xlabel('average price for quarter')
plt.xticks(range(0, 240, 10))

# The graph shows a small variance in the average price of properties across the 4 seasons.
# The average nightly price only varies from €220 in Q1 (quietest) to just under €230 in Q3 (busiest quarter).
# The seasonality of average price follows the seasonality of reviews/stays but is very subtle.
# It may be that relatively few Irish airbnb hosts are actively managing their calendar pricing.
# This may be the norm, or due to Covid and the time of year the data was scraped.

# Plot the monthly average price to see if this shows more variability
monthly_avg = df_calendar.groupby(df_calendar['date'].dt.strftime('%B'), sort=False)[['price']].mean()
monthly_avg.plot(kind='barh', figsize=(12, 7))
plt.xlabel('monthly average price')
plt.xticks(range(0, 240, 10))
plt.show()

# Monthly average price shows that there is variability in the calendar pricing with relatively
# slightly higher average prices in busier months but the range is narrow.

# Review Variability in Pricing by Day of Week

# Create weekday column
df_calendar['weekday'] = df_calendar['date'].dt.day_name()
# Create a list for the days of the week to use for reindexing
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# Create a sub dataframe with just weekday and price columns
sub_df = df_calendar[['weekday', 'price']]
# Groupby to get the average price by weekday and reindex using the days list
sub_df = df_calendar.groupby(['weekday']).mean().reindex(days)
# Drop 'listing_id' being the old index
sub_df.drop('listing_id', axis=1, inplace=True)

# Plot the average price by weekday
sub_df.plot(figsize=(12, 7))
ticks = list(range(0, 7, 1))
labels = "Mon Tues Weds Thurs Fri Sat Sun".split()
plt.xticks(ticks, labels)

# This shows that the average price by weekday peaks on the weekend, with Saturday the most expensive,
# followed by Friday and then Sunday. Prices are lowest on Monday to Wednesday and show a small increase
# on Thursday. This would suggest that stays between 1-4 nights over the weekend are the most popular as
# might be expected.

# Next look at occupancy using the listings which were active throughout 2019, are prices in a narrow
# range but occupancy is more variable?

# Use a groupby to get the count of reviews by listing_id by quarter in 2019
grp_reviews19_listing = df_reviews_2019.groupby(['listing_id', 'quarter_new']).agg({'listing_id': ['count']}).copy()

# to flatten the multi-level columns
grp_reviews19_listing.reset_index(inplace=True)

print(grp_reviews19_listing.head())

# Change quarter column to string dtype
grp_reviews19_listing['quarter_new'] = grp_reviews19_listing['quarter_new'].astype('str')

# Pivot to have listing_id in one column and each quarter be reflected in a column
grp_reviews19_listing = pd.pivot_table(grp_reviews19_listing, values='listing_id_count', index='listing_id',
                                       columns='quarter_new', aggfunc=np.sum)

# Convert the listing_id from an index to a column
grp_reviews19_listing.reset_index(level='listing_id', inplace=True)

# Check
grp_reviews19_listing.info()

# Use msno matrix to see positional information for the missing values
msno.matrix(grp_reviews19_listing)

# Merge df_listings with grp_reviews19_listing to bring the quarterly review count columns in to df_listings
df_listings = df_listings.merge(grp_reviews19_listing[['listing_id', '2019Q1', '2019Q2', '2019Q3', '2019Q4']],
                                how='left', left_on='id', right_on='listing_id').drop(columns=['listing_id'])

# Create a subset of the listings dataframe that contains listings active throughout 2019 to look at the data
# with a full year of occupancy pre Covid.  Include listings with a review in 2019, first review on or before
# 01/01/19 and last review on or after 31/12/2019.
df_listings2019 = df_listings.loc[(df_listings['2019Q1'].notnull()) | (df_listings['2019Q2'].notnull()) |
                                  (df_listings['2019Q3'].notnull()) | (df_listings['2019Q4'].notnull()) &
                                  (df_listings['first_review'] <= '2019-01-01') &
                                  (df_listings['last_review'] >= '2019-12-31'), :].copy()

# Replace null values with 0s in quarter (review count columns)
df_listings2019['2019Q1'] = df_listings2019['2019Q1'].fillna(0)
df_listings2019['2019Q2'] = df_listings2019['2019Q2'].fillna(0)
df_listings2019['2019Q3'] = df_listings2019['2019Q3'].fillna(0)
df_listings2019['2019Q4'] = df_listings2019['2019Q4'].fillna(0)

# Create a column with total reviews in 2019 for each listing
df_listings2019['reviews_2019'] = df_listings2019['2019Q1'] + df_listings2019['2019Q2'] + \
                                  df_listings2019['2019Q3'] + df_listings2019['2019Q4']

# Based on an assumed 50% review rate as is used by Inside Airbnb themselves, the number of stays is
# double the number of reviews.
# For occupancy, an assumption also needs to be made about average length of stay (ALOS). Inside Airbnb uses
# 3.0, however they have used 3 in any region where no public statements were made by Airbnb about average stays,
# so this was not an informed assumption.
# # Looking at data for Ireland and lengths of stay for trips in Ireland by Irish residents and overseas
# visitors, I calculate a weighted average ALOS of 3.4 days (2.5 days for domestic trips and 6.5 days divided
# between 1.5 locations for overseas visitors).
# Cap occupancy at 70% as otherwise values arise > 100%.  This 70% (255.5 days) cap is used by Inside Airbnb
# on the basis that it is a relatively high, but reasonable number for a highly occupied "hotel".
# It controls for listings with a very high review rate etc.

# create a list of conditions
conditions = \
    [(df_listings2019['minimum_nights'] > 3) &
     ((df_listings2019['minimum_nights'] * df_listings2019['reviews_2019'] * 2) <= 255.5),
     (df_listings2019['minimum_nights'] <= 3) & ((3.4 * df_listings2019['reviews_2019'] * 2) <= 255.5),
     ((df_listings2019['minimum_nights'] * df_listings2019['reviews_2019'] * 2) > 255.5),
     ((3.4 * df_listings2019['reviews_2019'] * 2) > 255.5)]

# create a list of the values to assign for each condition
values = [(df_listings2019['minimum_nights'] * df_listings2019['reviews_2019'] * 2),
          (3.4 * df_listings2019['reviews_2019'] * 2), 255.5, 255.5]

# create a new column and use np.select to assign values to it using our lists as arguments
df_listings2019['occupancy_2019'] = np.select(conditions, values)

# create a revenue (in thousands) column using the new occupancy column values multiplied by the price
df_listings2019['revenue_2019'] = round((df_listings2019['occupancy_2019'] * df_listings2019['price'])/1000, 1)

print(df_listings2019.head(5))

# create a revenue (in thousands) column using the new occupancy column values multiplied by the price
df_listings2019['revenue_2019'] = round((df_listings2019['occupancy_2019'] * df_listings2019['price']) / 1000, 1)

df_listings2019.head(5)






