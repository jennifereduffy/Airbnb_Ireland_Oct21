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
import re
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from pandas.api.types import CategoricalDtype

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

# Rename columns
grp_reviews19_listing.columns = ['listing_id', 'quarter_new', 'listing_id_count']

print(grp_reviews19_listing.head())

# Change quarter column to string dtype
grp_reviews19_listing['quarter_new'] = grp_reviews19_listing['quarter_new'].astype('str')

grp_reviews19_listing.info()

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

# Check hw many of each room_type in df_listings2019
df_listings2019['room_type'].value_counts()

# Create a copy of df_listings2019 with just entire homes for boxplot
df_entire2019 = df_listings2019.loc[df_listings2019['room_type'] == 'Entire home/apt'].copy()

# Create boxplot to show overall distribution of estimated annual occupancy in days for listings active throughout 2019
# by NUTS3 Region
plt.figure(figsize=(15, 15))
sns.color_palette("flare", as_cmap=True)
sns.boxplot(x=df_entire2019['NUTS3_Region'], y=df_entire2019['occupancy_2019'])
plt.show()

# Although some listings may not be available all year round, and their low occupancy may not be a reflection of
# low demand, this graph suggests the high occupancy levels which are possible.  In Dublin 50% of the entire
# home listings were occupied for over 140 days per year in 2019 and for many listings the occupancy was capped
# out at 70% by the maximum assumption made.  Compared to the relatively narrow range for price, there may be
# more scope to increase revenue by trying to maximise occupancy.

# Further review and clean main listings dataset with a view to machine learning

df_entire2019.info()

# Fill null values in host_since field with value in first_review field
df_entire2019.loc[df_entire2019['host_since'].isna(), 'host_since'] = df_entire2019['first_review']

# Create duration_as_host field based on days between date data was scraped and 'host_since'
dt1 = pd.to_datetime('22-10-2021')
df_entire2019['duration_as_host'] = round((dt1 - df_entire2019['host_since']).dt.days, 0)

# Convert dtype to integer
df_entire2019['duration_as_host'] = df_entire2019['duration_as_host'].astype('int')

# Eliminate some columns that won't be used in any model to predict occupancy to reduce noise at this stage.
# Some of these were used to derive the current subset of df_listings and are now implicit or captured by other
# columns.  Others such as description and latitude/longitude are not suitable for machine learning but may be
# reintroduced later.

columns_to_drop = ['description', 'latitude', 'longitude', 'room_type', 'number_of_reviews_ltm', 'first_review',
                   'last_review', '2019Q1', '2019Q2', '2019Q3', '2019Q4', 'host_since']
df_entire2019.drop(columns=columns_to_drop, inplace=True)

# change id column to string as the values are listing identifiers not integers
df_entire2019['id'] = df_entire2019['id'].astype('str')

# Convert values in host_about to either 1 or 0 rather than narrative or null value i.e. they have completed
# this field or not as this may add a personal touch that may make a difference
df_entire2019.host_about.fillna(0, inplace=True)
df_entire2019['host_about'] = df_entire2019['host_about'].apply(lambda v: 1 if v != 0 else 0)

df_entire2019['host_is_superhost'].value_counts()

# Convert to numerical column, f = False = 0, t = True = 1
d = {'f': 0, 't': 1}
df_entire2019['host_is_superhost'] = df_entire2019['host_is_superhost'].map(d).\
    fillna(df_entire2019['host_is_superhost'])

# 'bedrooms' column has null values but accommodates column does not.
# Use groupby to see the relationships between how many people a listing accommodates and how many
# bedrooms for those rows which do not have null value under bedrooms.
df_impute = df_entire2019.loc[df_entire2019['bedrooms'].notna(), :]
df_impute.groupby('bedrooms')[['accommodates']].median()

# We can see that using the median, the rule of thumb looks to be that each bedroom accommodates 2 people and
# there looks to be a cap on accommodates field of 16.

# Check if there is a max value of 16 in the accommodates field
print(df_listings['accommodates'].max())

# In August 2020, Airbnb wrote "Today we’re announcing a global ban on all parties and events at Airbnb listings,
# including a cap on occupancy at 16...."

# Fill missing values in the bedrooms field with the value in the accommodates field divided by 2
df_entire2019['bedrooms'].fillna(value=(df_entire2019['accommodates']//2), inplace=True)

# Check unique entries in bathrooms_text
df_entire2019['bathrooms_text'].unique()

# Replace Half-bath with 0.5 baths
df_entire2019['bathrooms_text'].replace({'Half-bath': '0.5 baths'}, inplace=True)

# Check how many listings have '0 baths'
df_entire2019['bathrooms_text'].value_counts()

# Take a look at the 0 bath rows:
print(df_entire2019.loc[df_entire2019['bathrooms_text'] == "0 baths", ['id', 'property_type', 'bedrooms']])

# Above some of the 0 baths listings look like errors.  Others look like they are camping & similar property types.
# For the camping type properties, keep 0 baths.

# Where bathrooms_text has 0 baths and the property_type is in list1, replace with 'MissingData'.

list1 = ['Entire residential home', 'Entire rental unit', 'Entire cottage', 'Entire townhouse',
         'Entire guest suite', 'Entire bed and breakfast', 'Entire bungalow']
for i in list1:
    df_entire2019.loc[(df_entire2019['bathrooms_text'] == "0 baths") &
                      (df_entire2019['property_type'] == i), 'bathrooms_text'] = 'MissingData'

# Use regex to isolate the numeric element of bathrooms_text.  Test on examples first
example1 = '3.5 baths'
print(re.findall(r'\s*[a-zA-Z]+', example1))

example2 = '1 bath'
print(re.findall(r'\s*[a-zA-Z]+', example2))

# Use list comprehension with regex tested above to extract the numerical element of bathrooms_text and append
# it to a new list called bathrooms_new
bathrooms_new = []
for row in df_entire2019['bathrooms_text']:
    if row == 'MissingData':
        number = 'MissingData'
    else:
        str1 = re.findall(r'\s*[a-zA-Z]+', row)
        number = row.replace(str1[0], "")
    bathrooms_new.append(number)

# Create a new column in df_entire2019 with just the numeric part of bathrooms_text and call it bathrooms_new
df_entire2019['bathrooms_new'] = bathrooms_new

df_entire2019['bathrooms_new'].value_counts()

# Drop the rows with MissingData in the bathrooms_new field
df_entire2019.drop(df_entire2019[df_entire2019['bathrooms_new'] == 'MissingData'].index, inplace=True)

# Now drop the no longer required bathrooms_text field and the beds field
df_entire2019.drop(['beds', 'bathrooms_text'], axis=1, inplace=True)

df_entire2019['property_type'].value_counts()

# Create a list of the camping style properties to drop the listings with these types
values = ["Hut", "Camper/RV", "Campsite", "Yurt", "Treehouse", "Tent", "Train", "Cave", "Shepherd's hut",
          "Bus", "Tipi"]
# Keep rows that don't contain any value in the list
df_entire2019 = df_entire2019[df_entire2019['property_type'].isin(values) == False]

# Take a look at values in maximum nights column, majority are set to max value allowable of 1125.
# It is low maximum nights that could have an influence on occupancy so I'm interested in the low
# end of the range.

# Use bins in histogram to see how the data is distributed across listings with max nights up to 90 nights
bins_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 30, 32, 34,
             36, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
plt.hist(df_entire2019['maximum_nights'], bins=bins_list)

# If maximum_nights > 90, change to 90 as a new imposed cap on the basis that whether the max nights is 90
# or the max (1125),this should not make any notable difference
df_entire2019.loc[df_entire2019['maximum_nights'] > 90, 'maximum_nights'] = 90

# Create a scatter plot to see if there is a relationship between max nights and occupancy
x = df_entire2019['maximum_nights']
y = df_entire2019['occupancy_2019']

plt.scatter(x, y)
plt.xlabel('Maximum Nights Stay')
plt.ylabel('Occupancy')
plt.show()

# Difficult to see relationship in scatter plot, use Linear Regression to calculate correlation coefficient

# create X and y
feature_cols = ['maximum_nights']
X = df_entire2019[feature_cols]
y = df_entire2019.occupancy_2019

# follow the usual sklearn pattern: import, instantiate, fit
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print("Intercept is", lm.intercept_)
print("Max nights coefficient is", lm.coef_)

# This suggest that there is some positive correlation between maximum nights and occupancy so keep max nights field.

# Next have a look at superhost and other reputational capital fields to see what relationships if any they have to
# occupancy (and price)

# Key criteria to become a Superhost below
# Have had min 10 booked guest trips OR successfully completed 3 long term reservations that total min 100 nights
# Have maintained a 50% review rate or higher.
# Have maintained a 90% response rate or higher
# Zero cancellations, except for situations that fall under Airbnb's Extenuating Circumstances policy.
# Have maintained an overall 4.8/5 rating
# Note exceptions have been made to superhost criteria due to Covid. Currently they don't have to meet the 10 bookings
# in past year criteria.

# Take a look at the listings with nan in host_is_superhost and look at most relevant other fields taking in to account
# superhost criteria
print(df_entire2019.loc[df_entire2019['host_is_superhost'].isna(),
                        ['id', 'review_scores_rating', 'host_response_rate', 'number_of_reviews', 'reviews_2019']])

# Check Airbnb site to see if the first two listings are superhosts given high review_scores_rating and complete the nan
# values in host_is_superhost accordingly
# https://www.airbnb.com/rooms/21722118 - YES
# https://www.airbnb.com/rooms/24632132 - NO

df_entire2019.loc[df_entire2019['id'] == 2172118, 'host_is_superhost'] = 1
df_entire2019['host_is_superhost'].fillna(value=0, inplace=True)

# Create a scatter plot of price and occupancy showing superhost and non superhost in different colours
sns.relplot(x='occupancy_2019', y='price', hue='host_is_superhost', data=df_entire2019)

# The scatter plot suggests superhosts have higher occupancy but is less clear regarding price.

# There are listings with zero bedrooms.  Review before boxplots.
print(df_entire2019.loc[df_entire2019['bedrooms'] == 0, :])

# Check the description field to see if it contains the number of bedrooms
print(df_listings.loc[df_listings['id'] == 34541926, 'description'])
print(df_listings.loc[df_listings['id'] == 37377666, 'description'])
print(df_listings.loc[df_listings['id'] == 18328816, 'description'])

# Override 0 values in bedrooms field with bedroom count per description.  For Studios, use 1.
df_entire2019['bedrooms'] = df_entire2019['bedrooms'].astype(int)
df_entire2019.loc[df_entire2019['id'] == '34541926', 'bedrooms'] = 1
df_entire2019.loc[df_entire2019['id'] == '37377666', 'bedrooms'] = 4
df_entire2019.loc[df_entire2019['id'] == '18328816', 'bedrooms'] = 1

# Create a sub df for a clearer boxplot including listings with 3 or fewer bedrooms and price under 1750
df_temp2 = df_entire2019[((df_entire2019['bedrooms'] < 4) & (df_entire2019['price'] < 1750))].copy()
# Check count of listings in sub df, still is the majority
print(df_temp2.shape)

# Create a boxplot comparing the 2019 prices for superhosts Vs other hosts for this subset
plt.figure(figsize=(8, 5))
sns.boxplot(x='bedrooms', y='price', hue='host_is_superhost', data=df_temp2, palette='rainbow')
plt.title("Price for Superhosts Vs Non-Superhosts")
plt.show()

# Create a boxplot comparing the 2019 occupancy for superhosts V other hosts for this subset
plt.figure(figsize=(8, 5))
sns.boxplot(x='bedrooms', y='occupancy_2019', hue='host_is_superhost', data=df_temp2, palette='rainbow')
plt.title("Occupancy for Superhosts Vs Non-Superhosts")
plt.show()

# The 1st boxplot above indicates that for similar sized properties, the median price for superhosts is slightly lower
# than for hosts who have not achieved superhost status. This surprised me but some web searching indicates that this
# is generally the case globally.
# The 2nd boxplot indicates that median occupancy in 2019 for superhosts was much higher than for non superhosts.
# It may be that superhosts have a better understanding of their market and getting pricing right to optimise price
# & occupancy for maximum revenue. Some hosts internationally use smart pricing software, and it is likely the case that
# some Irish hosts are also using it.

# Plot the distribution of the overall review score for df_entire2019.
plt.figure(figsize=(12, 6))
sns.displot(data=df_entire2019, x='review_scores_rating')
plt.xlim(0, 5)
plt.show()

# We can see that very few listings have an overall review score below 4, and the majority look to be close to the
# superhost level of 4.8.

# Check To see if there is a difference between superhost verus non superhosts who have a rating of 4.8+
# Create sub df with only listings with review scores rating >= 4.8
df_temp3 = df_temp2[(df_temp2['review_scores_rating'] >= 4.8)].copy()

# Create a boxplot comparing the 2019 prices for superhosts V other hosts with 4.8+ rating for this subset
plt.figure(figsize=(8, 5))
sns.boxplot(x='bedrooms', y='price', hue='host_is_superhost', data=df_temp3, palette='rainbow')
plt.title("Price for Superhosts Vs Non-Superhosts")
plt.show()

# Create a boxplot comparing the 2019 occupancy for superhosts V other hosts with 4.8+ rating for this subset
plt.figure(figsize=(8, 5))
sns.boxplot(x='bedrooms', y='occupancy_2019', hue='host_is_superhost', data=df_temp3, palette='rainbow')
plt.title("Occupancy for Superhosts Vs Non-Superhosts")
plt.show()

# Above boxplots still indicate it is advantageous for occupancy to be a superhost versus a non superhost with a high
# review rating

# There are 21 null values in all of the underlying review_scores fields i.e. review_score_accuracy,
# review_scores_cleanliness etc.  Check the nulls are all in the same rows.
print(df_entire2019.loc[df_entire2019['review_scores_accuracy'].isna(),
                        ['review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                         'review_scores_location', 'review_scores_value']])

# Drop rows where review_scores_accuracy values are null in a sub df
df_temp3 = df_entire2019[df_entire2019['review_scores_accuracy'].notna()]

# Create a sub dataframe with just the 7 review score columns and the occupancy_2019 column for input to
# correlation matrix
df_temp4 = df_temp3[['occupancy_2019', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                     'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                     'review_scores_value']]

# Plot heatmap showing correlation between columns in df_temp4
corrMatrix = df_temp4.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# Most correlated to occupancy are the cleanliness and value scores, followed by the accuracy score.
# Accuracy, cleanliness, and values all have a correlation coefficient > 0.7 with the overall review score.

# All the review scores are positively correlated with host_is_superhost. Based on the above,
# and to reduce noise in my dataset, I am going to keep host_is_superhost and review_scores_rating and drop the other
# underlying review fields. Due to the distribution of the overall review score with most scores between
# 4 & 5 out of 5, I am going to create bins and an ordered categorical column to better reflect this i.e. 4/5
# is a bad review not an 80% grade.

# Check min value in review_scores_rating column
print(df_entire2019['review_scores_rating'].min())
# Check how many listings have a 0.0 rating
print(df_entire2019.loc[df_entire2019['review_scores_rating'] == 0, 'id'].count())

# Use pd.cut to create bins with self-defined bin ranges on overall review_scores_rating
cut_labels = ['<=3', '<=4', '<=4.2', '<=4.4', '<=4.6', '<=4.8', '<=5.0']
cut_bins = [0.0, 3.0, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]
df_entire2019['cut_review_rating'] = pd.cut(df_entire2019['review_scores_rating'], bins=cut_bins, labels=cut_labels,
                                            include_lowest=True)

# Change new cut_review_rating field to categorical, and order from low to high
rating_order = CategoricalDtype(categories=['<=3', '<=4', '<=4.2', '<=4.4', '<=4.6', '<=4.8', '<=5.0'], ordered=True)
df_entire2019['cut_review_rating'] = df_entire2019['cut_review_rating'].astype(rating_order)
df_entire2019['cut_review_rating'].describe()

# Now drop the no longer required bathrooms_text field and the beds field
df_entire2019.drop(['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                    'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                    'review_scores_value'], axis=1, inplace=True)

# Check
df_entire2019.info()

# Also drop host_reponse_time, host_reponse_rate, host_acceptance_rate as many null values and are covered to a degree
# in host_is_superhost
df_entire2019.drop(['host_response_time', 'host_response_rate', 'host_acceptance_rate'], axis=1, inplace=True)

# Drop accommodates field as it is highly correlated to bedrooms field (seen earlier)
df_entire2019.drop(['accommodates'], axis=1, inplace=True)

# Drop reviews_2019 and revenue_2019 as these are highly correlated to occupancy_2019
df_entire2019.drop(['reviews_2019', 'revenue_2019'], axis=1, inplace=True)

df_entire2019['availability_365'].describe()

# Check relationship between availability_365 and target variable occupancy

# Create a scatter plot to see if there is a relationship between availability (forward looking) and
# occupancy (historical)
x = df_entire2019['availability_365']
y = df_entire2019['occupancy_2019']

plt.scatter(x, y)
plt.xlabel('Availability Next 365')
plt.ylabel('Occupancy 2019')
plt.show()

# Difficult to see relationship in scatter plot, use Linear Regression to calculate correlation coefficient
# create X and y
feature_cols = ['availability_365']
X = df_entire2019[feature_cols]
y = df_entire2019.occupancy_2019

# follow the usual sklearn pattern: import, instantiate, fit
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print("Intercept is", lm.intercept_)
print("Availability coefficient is", lm.coef_)

df_entire2019['NUTS3_Region'].value_counts()

# There are 32 different region_parent_names and 8 NUTS3_Region names.  I don't think it makes sense to hot encode
# 32 different regions which will add 32 additonal columns to the dataset. It also may not add insight to the
# model given the sparsity of listings in some regions.  From my knowledge of Ireland and tourism, the NUTS 3 Regions
# do not group region_parents together in a way that makes sense for tourism so I have reorganised them in to 12 groups.

df_regions_reconfig = \
    pd.read_csv(r"C:\Users\jenni\Documents\Datasets\Ireland Oct 21 Airbnb\Failte Ireland\Regions_Reconfigured.csv")

print(df_regions_reconfig)






