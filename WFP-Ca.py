import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#DataCleaning
# Loading the dataset 
df = pd.read_csv(
    r"C:\Users\91701\Downloads\wfpvam_foodprices.csv",
    dtype={'adm1_id': str},
    low_memory=False
)

print("Shape of the dataset:", df.shape)
print("First few rows:")
print(df.head())

# missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Dropping rows with missing prices
df = df.dropna(subset=['mp_price'])
df = df.drop_duplicates()

# date column
df['date'] = pd.to_datetime(df['mp_year'].astype(str) + '-' + df['mp_month'].astype(str) + '-01')

# Handling unknown values
# Fill missing 'adm1_name' with 'adm0_name' 
df['adm1_name'] = df['adm1_name'].fillna(df['adm0_name'])

# Dropping mp_commoditysource (NAN for every row)
df = df.drop(columns=['mp_commoditysource'])

# Rename columns for better clarity
df.rename(columns={'adm0_id':'country_id',
                   'adm0_name':'country',
                   'adm1_id':'province_id',
                   'adm1_name':'province',
                   'mkt_id':'city_id',
                   'mkt_name':'city',
                   'cm_id':'food_id',
                   'cm_name':'food',
                   'mp_month':'month',
                   'mp_year':'year',
                   'mp_price':'price',
                   'um_name':'unit',
                   'cur_name':'currency',
                   'cur_id':'currency_id',
                   'um_id':'unit_id',
                   'pt_name':'purchase_type',
                   'pt_id':'purchase_type_id'},
          inplace=True)

print("\nAfter renaming columns:")
print(df.columns)

print("\nData after cleaning:")
print(df.head())
print(df.info())
print(df.isnull().sum())

#EDA and Visulization ---------------------------------------------------------------
# Number of rows for each country
country_unique, country_freq = np.unique(df['country'], return_counts=True)
d1 = pd.DataFrame({'Country': country_unique, 'RowCount': country_freq})
d1 = d1.sort_values(by='RowCount', ascending=False)

# Plot styling
plt.figure(figsize=(22, 10))
sns.set_style("darkgrid")

barplot = sns.barplot(data=d1, x='Country', y='RowCount', palette='viridis')

# Title and labels
plt.title('Number of Data Rows per Country', fontsize=22, weight='bold')
plt.xlabel('Country', fontsize=16)
plt.ylabel('Row Count', fontsize=16)

# Rotate x-axis labels
plt.xticks(rotation=60, ha='right', fontsize=10)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

#plt.style.use('seaborn-v0_8')  # Modern equivalent of 'seaborn' style
plt.style.use('dark_background')
sns.set_style("darkgrid")
sns.set_palette("husl")

#print("\nBasic statistics for numerical columns:")
#print(df.describe())

# Created price categories
bins = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
labels = ['<$1', '$1-2', '$2-5', '$5-10', '$10-20', 
          '$20-50', '$50-100', '$100-200', '$200-500', '$500+']

df['price_bin'] = pd.cut(df['price'], bins=bins, labels=labels)

plt.figure(figsize=(10,5))
df['price_bin'].value_counts().sort_index().plot(
    kind='bar', 
    color='darkorange',
    edgecolor='black'
)
plt.title('Food Prices by Price Range')
plt.xlabel('Price Ranges (USD)')
plt.ylabel('Number of Products')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Price Trends Over Time (Median prices)
plt.figure(figsize=(14, 6))
df.groupby('date')['price'].median().plot()
plt.title('Median Food Price Trend Over Time')
plt.ylabel('Price')
plt.xlabel('Date')
plt.grid(True)
plt.show()


#-------------------------------------------------------------------------------------------------------------
#PREPARING DATA FOR FOUR COUNTRIES(TURKEY,IRAQ,IRAN,SYRIA)
#Turkey
turkey = df.loc[df['country'] == 'Turkey' , ['country','city','purchase_type','food','unit','price','month','year']]
#Iran
iran = df.loc[df['country'].str.contains('Iran') , ['country','city','purchase_type','food','unit','price','month','year']]
#Iraq
iraq = df.loc[df['country'].str.contains('Iraq') , ['country','city','purchase_type','food','unit','price','month','year']]
#Syria
syria = df.loc[df['country'].str.contains('Syria') , ['country','city','purchase_type','food','unit','price','month','year']]
#changing_price_to_USD
exchange_rates = {
    'Turkey': 1 / 27.0,   # 1 TRY ≈ 0.037 USD
    'Iran':   1 / 42000,  # 1 IRR ≈ 0.000024 USD
    'Iraq':   1 / 1300,   # 1 IQD ≈ 0.00077 USD
    'Syria':  1 / 12500   # 1 SYP ≈ 0.00008 USD
}

# Add new column 'price_usd' with converted values
turkey["price_usd"] = turkey["price"] * exchange_rates['Turkey']
iran["price_usd"]   = iran["price"]   * exchange_rates['Iran']
iraq["price_usd"]   = iraq["price"]   * exchange_rates['Iraq']
syria["price_usd"]  = syria["price"]  * exchange_rates['Syria']

#turkey_top10_expensive
turkey_top10 = (
    turkey.groupby("food")
    .agg({
        "price": "mean",  # Take the average of the price
        "unit": lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan  # Most common unit
    })
    .sort_values(by="price", ascending=False)
    .head(10)
)

# Create the pie chart
plt.figure(figsize=(14, 14))

# Custom colors for better visual distinction
colors = plt.cm.Paired(range(len(turkey_top10)))

# Create the pie chart with percentage labels
wedges, texts, autotexts = plt.pie(
    turkey_top10['price'],
    labels=turkey_top10.index,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.85,
    textprops={'fontsize': 10}
)
plt.axis('equal')
plt.title('Top 10 Most Expensive Items in Turkey\n(Average Price)', pad=20, fontsize=14)
legend_labels = [f"{food} (${price:.2f}/{unit})" 
                for food, price, unit in zip(turkey_top10.index, turkey_top10['price'], turkey_top10['unit'])]
plt.legend(wedges, legend_labels,title="Food Items",loc="upper left",bbox_to_anchor=(1.05, 1.05),       borderaxespad=0.0
)
plt.tight_layout()
plt.show()

#Histplot to compare and find outliers---------------------------------------------------------------

#Lentils Price Distribution in Turkey, Iraq, Syria
# Turkey - Lentils Price Distribution
turkey_lentils= turkey[turkey['food'].isin(['Lentils - Retail'])]
plt.figure(figsize=(12, 5))
sns.histplot(turkey_lentils["price_usd"], color='r', kde=True)
plt.title("Distribution of Lentils Prices in Turkey")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Iraq - Lentils Price Distribution
iraq_lentils = iraq[iraq['food'].isin(['Lentils - Retail'])]
plt.figure(figsize=(12, 5))
sns.histplot(iraq_lentils["price_usd"], color='b', kde=True)
plt.title("Distribution of Lentils Prices in Iraq")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Syria - Lentils Price Distribution
syria_lentils = syria[syria['food'].isin(['Lentils - Retail'])]
plt.figure(figsize=(12, 5))
sns.histplot(syria_lentils["price_usd"], color='y', kde=True)
plt.title("Distribution of Lentils Prices in Syria")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



#Rice Price Distribution in Turkey, Iraq, Syria

# Turkey - RIce Price Distribution
turkey_rice = turkey[turkey['food'].isin(['Rice - Retail'])]
plt.figure(figsize=(12, 5))
sns.histplot(turkey_rice["price_usd"], color='r', kde=True)
plt.title("Distribution of Rice Prices in Turkey")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Iraq - Rice Price Distribution
iraq_rice = iraq[iraq['food'].isin(['Rice - Retail'])]
plt.figure(figsize=(12, 5))
sns.histplot(iraq_rice["price_usd"], color='b', kde=True)
plt.title("Distribution of Rice Prices in Iraq")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Syria - Rice Price Distribution
syria_rice = syria[syria['food'].isin(['Rice - Retail'])]
plt.figure(figsize=(12, 5))
sns.histplot(syria_rice["price_usd"], color='y', kde=True)
plt.title("Distribution of Rice Prices in Syria")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# Turkey - Rice Price Scatter Plot
plt.figure(figsize=(12, 5))
turkey_sugar = turkey[turkey['food'].isin(['Sugar - Retail'])]
sns.scatterplot(x=turkey_sugar.index, y=turkey_sugar["price_usd"], color='r')
plt.title("Scatter Plot of Rice Prices in Turkey")
plt.xlabel("Index")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.show()

# Iraq - Rice Price Scatter Plot
plt.figure(figsize=(12, 5))
iraq_sugar = iraq[iraq['food'].isin(['Sugar - Retail'])]
sns.scatterplot(x=iraq_sugar.index, y=iraq_sugar["price_usd"], color='b')
plt.title("Scatter Plot of Sugar Prices in Iraq")
plt.xlabel("Index")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.show()

# Syria - Rice Price Scatter Plot
plt.figure(figsize=(12, 5))
syria_sugar = syria[syria['food'].isin(['Sugar - Retail'])]
sns.scatterplot(x=syria_sugar.index, y=syria_sugar["price_usd"], color='y')
plt.title("Scatter Plot of Sugar Prices in Syria")
plt.xlabel("Index")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.show()


# Price Trends Over Time (Median prices in USD)
for df in [turkey, iran, iraq, syria]:
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
plt.figure(figsize=(14, 6))
sns.lineplot(data=turkey, x='date', y='price_usd', label='Turkey', color='red')
sns.lineplot(data=iran, x='date', y='price_usd', label='Iran', color='blue')
sns.lineplot(data=iraq, x='date', y='price_usd', label='Iraq', color='green')
sns.lineplot(data=syria, x='date', y='price_usd', label='Syria', color='orange')
plt.title('Food Price Trends Over Time (USD)')
plt.ylabel('Median Price (USD)')
plt.xlabel('Year')
plt.legend()
plt.grid(True)
plt.show()


# Minimum wage data (USD/hour)
min_wages = {
    'Turkey': 1.5,  # Approx. 40 TRY/hour ≈ 1.5 USD
    'Iran': 0.1,     # Approx. 420,000 IRR/hour ≈ 0.1 USD
    'Iraq': 0.3,     # Approx. 1,500 IQD/hour ≈ 0.3 USD
    'Syria': 0.05    # Approx. 6,250 SYP/hour ≈ 0.05 USD
}

# Calculate work hours needed to buy 1kg of rice
rice_prices = {
    'Turkey': turkey[turkey['food'] == 'Rice - Retail']['price_usd'].mean(),
    'Iran': iran[iran['food'] == 'Rice - Retail']['price_usd'].mean(),
    'Iraq': iraq[iraq['food'] == 'Rice - Retail']['price_usd'].mean(),
    'Syria': syria[syria['food'] == 'Rice - Retail']['price_usd'].mean()
}

hours_needed = {country: (rice_prices[country] / min_wages[country]) 
                for country in min_wages}

hours_df = pd.DataFrame(list(hours_needed.items()), columns=['Country', 'Hours_Needed'])

plt.figure(figsize=(8, 5))
sns.barplot(data=hours_df, x='Country', y='Hours_Needed', palette='rocket')
plt.title('Work Hours Needed to Buy 1kg of Rice')
plt.ylabel('Hours of Work')
plt.show()


# Compare average prices of staple foods
staple_foods = ['Rice - Retail', 'Lentils - Retail', 'Sugar - Retail']
avg_prices = []

for food in staple_foods:
    turkey_avg = turkey[turkey['food'] == food]['price_usd'].mean()
    iran_avg = iran[iran['food'] == food]['price_usd'].mean()
    iraq_avg = iraq[iraq['food'] == food]['price_usd'].mean()
    syria_avg = syria[syria['food'] == food]['price_usd'].mean()
    avg_prices.append([food, turkey_avg, iran_avg, iraq_avg, syria_avg])

avg_df = pd.DataFrame(avg_prices, columns=['Food', 'Turkey', 'Iran', 'Iraq', 'Syria'])
avg_df.set_index('Food', inplace=True)

plt.figure(figsize=(10, 6))
sns.heatmap(avg_df, annot=True, cmap='YlOrRd', fmt='.2f')
plt.title('Average Staple Food Prices (USD) Across Countries')
plt.show()
#------------------------------------------------------------------------------------------------
#Descriptive Statistics for Sugar,Lentil and Rice Price in Syria
syria_lentils.describe()
syria_sugar.describe()
syria_rice.describe()

#Finding Outlier Values for Lentil,Sugar and Rice prices in Syria
#IQR
q1 = syria_lentils['price'].quantile(.25)
print('Lentils'' q1 is {}'.format(q1))
q3 = syria_lentils['price'].quantile(.75)
print('Lentils'' q3 is {}'.format(q3))
IQR = q3- q1
lower_bound = q1 - 1.5 * IQR
upper_bound = q3 + 1.5 * IQR
# Filter the dataset to exclude the outliers
syria_lentils_v2 = syria_lentils.copy()  
syria_lentils_v2 = syria_lentils_v2[(syria_lentils_v2['price'] >= lower_bound) & (syria_lentils_v2['price'] <= upper_bound)]

# Box plot for the cleaned Syria Lentils prices
plt.figure(figsize=(12, 6))
syria_lentils_v2.boxplot(column=['price'], rot=45, patch_artist=True,  boxprops=dict(facecolor='skyblue', color='black'),whiskerprops=dict(color='orange', linewidth=1.5),  flierprops=dict(marker='o', color='red', markersize=7),  medianprops=dict(color='green', linewidth=2))
plt.title("Outlier Detection - Cleaned Data for Syria Lentils Prices", fontsize=16)
plt.xlabel("Price", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.yscale("log")  # Apply logarithmic scale if data is skewed
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
plt.tight_layout()
plt.show()

# Box plot for the original Syria Sugar prices
plt.figure(figsize=(12, 6))
syria_sugar.boxplot(column=['price'], rot=45, patch_artist=True,boxprops=dict(facecolor='lightcoral', color='black'))
plt.title("Outlier Detection - Original Syria Sugar Prices", fontsize=16)
plt.xlabel("Price", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
plt.tight_layout()
plt.show()


#z-score
mean_price = syria_sugar['price'].mean()
std_price = syria_sugar['price'].std()
# Calculate Z-scores
syria_sugar['z_score'] = (syria_sugar['price'] - mean_price) / std_price
z_threshold = 3
syria_sugar_z_outliers = syria_sugar[syria_sugar['z_score'].abs() > z_threshold]
print(f"Outliers detected based on Z-score:\n{syria_sugar_z_outliers}")
syria_sugar_cleaned = syria_sugar[syria_sugar['z_score'].abs() <= z_threshold]

# Box plot for the cleaned Syria Sugar prices
plt.figure(figsize=(12,6))
syria_sugar_cleaned.boxplot(column=['price'], rot=45)
plt.title("Box Plot for Outlier Detection - Cleaned Data")
plt.show()

#-------------------------------------------------------------------------------------------------------
#Correlation-matrix


#1- Correlation between food prices over time
plt.figure(figsize=(12, 10))
# Pivot to get median prices by date and country
price_pivot = df.pivot_table(values='price', index='date', columns='country', aggfunc='median')
# Calculate correlation matrix
corr_matrix = price_pivot.corr()
# Plot heatmap
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,linewidths=0.5)
plt.title('Correlation Between Food Prices Across Countries', pad=20)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#2- Correlation between different food items in Turkey
plt.figure(figsize=(14, 10))
# Select top 10 common foods in Turkey
top_foods = turkey['food'].value_counts().head(10).index.tolist()
turkey_top = turkey[turkey['food'].isin(top_foods)]
turkey_top['date'] = pd.to_datetime(turkey_top['year'].astype(str) + '-' + turkey_top['month'].astype(str) + '-01')
pivot_turkey = turkey_top.pivot_table(values='price_usd', index='date', columns='food', aggfunc='mean')
food_corr = pivot_turkey.corr()
# Plot heatmap
mask = np.triu(np.ones_like(food_corr, dtype=bool))
sns.heatmap(food_corr,annot=True,cmap='RdYlGn',center=0,vmin=-1, vmax=1,fmt='.2f')
plt.title('Correlation Between Food Prices in Turkey', pad=20)
plt.tight_layout()
plt.show()


#3- Correlation between countries for specific commodities
common_foods = ['Rice - Retail', 'Sugar - Retail', 'Lentils - Retail']
# Create a combined dataframe
combined = pd.concat([
    turkey.assign(country='Turkey'),
    iran.assign(country='Iran'),
    iraq.assign(country='Iraq'),
    syria.assign(country='Syria')
])
combined = combined[combined['food'].isin(common_foods)]
combined['date'] = pd.to_datetime(combined['year'].astype(str) + '-' + combined['month'].astype(str) + '-01')
pivot_combined = combined.pivot_table(values='price_usd', index=['date', 'food'], columns='country', aggfunc='median')
# Calculated correlation for each food
for food in common_foods:
    food_data = pivot_combined.xs(food, level='food')
    corr = food_data.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='YlOrRd', vmin=0,vmax=1,fmt='.2f')
    plt.title(f'Price Correlation for {food} Across Countries')
    plt.tight_layout()
    plt.show()