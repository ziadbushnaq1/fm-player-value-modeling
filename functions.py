import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re

def process_fm_data(filepath, league_rep_filepath, game_date_str='1/5/2035'):
    """
    Loads, cleans, and engineers features for a Football Manager player dataset.

    Args:
        filepath (str): Path to the HTML file with player data.
        league_rep_filepath (str): Path to the CSV/HTML file with the ordered league reputation list.
        game_date_str (str): The current in-game date as a string (e.g., '1/5/2035').

    Returns:
        pandas.DataFrame: A fully cleaned and processed DataFrame.
    """
    try:
        df = pd.read_html(filepath)[0].copy()
        df = df.iloc[1:]
    except (IOError, IndexError) as e:
        print(f"Error reading player file {filepath}: {e}")
        return None

    def parse_height(height_str):
        match = re.match(r"(\d+)'(\d+)", str(height_str))
        if match:
            feet, inches = map(int, match.groups())
            return feet * 12 + inches
        return np.nan

    def clean_value(value):
        if pd.isna(value) or value == 'Not for Sale': return np.nan
        value_str = str(value).replace('Â£', '').replace('£', '').strip()
        if not value_str: return np.nan
        def convert_suffix(val_str):
            val_str = val_str.strip()
            if 'M' in val_str: return float(val_str.replace('M', ''))
            if 'K' in val_str: return float(val_str.replace('K', '')) / 1000
            return float(val_str)
        if '-' in value_str:
            low, high = value_str.split('-')
            return (convert_suffix(low) + convert_suffix(high)) / 2.0
        else:
            return convert_suffix(value_str)

    def clean_fee(fee):

        if pd.isna(fee) or isinstance(fee, (int, float)): return fee
        fee_str = str(fee).replace('Â£', '').replace('£', '').strip()
        if not fee_str or fee_str in ['-', '- - -', 'Free']: return 0.0
        try:
            if 'M' in fee_str: return float(fee_str.replace('M', ''))
            elif 'K' in fee_str: return float(fee_str.replace('K', '')) / 1000
            else: return np.nan
        except ValueError: return np.nan
            
    def combine_apps(apps_str):
        if pd.isna(apps_str): return 0
        numbers = re.findall(r'\d+', str(apps_str))
        return sum(int(num) for num in numbers) if numbers else 0

    df['Height'] = df['Height'].apply(parse_height)
    df['Wage'] = df['Wage'].str.replace(r"[^\d.]", "", regex=True).replace('', np.nan).astype('Float64')
    df['Transfer Value'] = df['Transfer Value'].apply(clean_value)
    df['Last Trans. Fee'] = df['Last Trans. Fee'].apply(clean_fee)
    if 'Transfer Fees Received' in df.columns:
        df['Transfer Fees Received'] = df['Transfer Fees Received'].apply(clean_fee)
    df['Total Apps'] = df['Apps'].apply(combine_apps)
    df['Transfer_Status_bool'] = (df['Transfer Status'] != 'Not set').astype(int)
    df['Country'] = df['Based'].str.split('(').str[0].str.strip()

    df['Expires'] = pd.to_datetime(df['Expires'], errors='coerce')
    df['Begins'] = pd.to_datetime(df['Begins'], errors='coerce')
    current_game_date = pd.to_datetime(game_date_str)
    
    df['Days Until Expiry'] = (df['Expires'] - current_game_date).dt.days
    years_since_signing = (current_game_date - df['Begins']).dt.days / 365.25
    df['Age_at_Signing'] = df['Age'].astype(float) - years_since_signing
    df['Years_at_Club'] = (current_game_date - df['Begins']).dt.days / 365.25

    tier_1 = ['Slack', 'Casual', 'Temperamental', 'Spineless', 'Low Self-Belief', 'Easily Discouraged', 'Low Determination']
    tier_2 = ['Fickle', 'Mercenary', 'Unambitious', 'Unsporting', 'Realist']
    tier_3 = ['Balanced', 'Light-Hearted', 'Jovial', 'Very Loyal', 'Devoted', 'Loyal', 'Fairly Loyal', 'Honest', 'Sporting', 'Fairly Sporting']
    tier_4 = ['Perfectionist', 'Resolute', 'Professional', 'Fairly Professional', 'Iron Willed', 'Resilient', 'Spirited', 'Driven', 'Determined', 'Fairly Determined', 'Charismatic Leader', 'Born Leader', 'Leader', 'Very Ambitious', 'Fairly Ambitious', 'Ambitious']
    tier_5 = ['Model Professional']
    personality_tiers = [tier_1, tier_2, tier_3, tier_4, tier_5]
    personality_map = {p: i + 1 for i, tier in enumerate(personality_tiers) for p in tier}
    df['Personality_Tier'] = df['Personality'].map(personality_map)
    df['Personality_Tier'] = df['Personality_Tier'].fillna(0)
    
    try:
        league_rep_df = pd.read_html(league_rep_filepath)[0] 
        ordered_leagues = league_rep_df['Name'].tolist()
        num_leagues = len(ordered_leagues)
        league_rank_map = {league: num_leagues - i for i, league in enumerate(ordered_leagues)}
        df['Division_Rank'] = df['Division'].map(league_rank_map)
        df['Division_Rank']= df['Division_Rank'].fillna(0)
    except IOError:
        print(f"Warning: League reputation file not found at {league_rep_filepath}. Skipping division ranking.")
        df['Division_Rank'] = 0 
        
    country_dummies = pd.get_dummies(df['Country'], prefix='Country', dummy_na=True)
    df = pd.concat([df, country_dummies], axis=1)

    return df


def process_player_data(df):
    """
    Applies cleaning and feature transformation steps from the initial analysis
    to a player dataset.
    
    Args:
        df (pd.DataFrame): Raw dataframe for a specific position.
        
    Returns:
        pd.DataFrame: Cleaned and transformed dataframe.
    """
    df_clean = df.copy()
    
    financial_cols = ['Transfer Value', 'Wage', 'Last Trans. Fee']
    for col in financial_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
            
    if 'Av Rat' in df_clean.columns:
        df_clean['Av Rat'] = pd.to_numeric(df_clean['Av Rat'], errors='coerce')
        df_clean['Av Rat'] = df_clean['Av Rat'].fillna(6.7)

    xg_cols = ['xGP/90', 'xGP']
    
    for col in xg_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(0)

    date_derived_features = ['Days Until Expiry', 'Age_at_Signing', 'Years_at_Club']
    for col in date_derived_features:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    if 'Transfer Value' in df_clean.columns:
        df_clean['Value_log'] = np.log1p(df_clean['Transfer Value'])
        
    if 'Wage' in df_clean.columns:
        df_clean['Wage_log'] = np.log1p(df_clean['Wage'])
        
    if 'Last Trans. Fee' in df_clean.columns:
        df_clean['lastfee_log'] = np.log1p(df_clean['Last Trans. Fee'])

    return df_clean

# cb_df_cleaned = process_player_data(cb_df)
# fb_df_cleaned = process_player_data(fb_df)
# dm_df_cleaned = process_player_data(dm_df)

# Check the results
# print(cb_df_cleaned[['Name', 'Value_log', 'Wage_log']].head())

def train_valuation_model(df):
    """
    Trains a Random Forest model to predict Intrinsic Player Value.
    Excludes Wage, Contract, and Nationality to find undervalued targets.
    """
    
    excluded_cols = [
        # Targets & Money (Prevent Leakage)
        'Transfer Value', 'Value_log', 
        'Wage', 'Wage_log', 
        'Last Trans. Fee', 'lastfee_log', 
        'Transfer Fees Received', 'totalfees_log',
        
        # Identifiers & Text
        'Name',
        'Based', 'Division', 'Nationality', 'Country', # Text columns
        'Personality', 'Position', # Text metadata
        
        # Contract Details 
        'Expires', 'Days Until Expiry', 'Begins', 
        'Age_at_Signing', 'Years_at_Club',
        
        # Market Status 
        'Transfer Status', 'Transfer_Status_bool', 
        'Rec', 'Inf',
        
        'WR'
    ]
    
    country_cols = [c for c in df.columns if c.startswith('Country_')]
    excluded_cols.extend(country_cols)

    feature_cols = [c for c in df.columns 
                    if c not in excluded_cols 
                    and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"Training on {len(feature_cols)} features...")
    print(f"Top 10 Features included: {feature_cols[:10]}") 
    
    X = df[feature_cols]
    y = df['Value_log'] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred_log = rf.predict(X_test)
    
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_log))
    r2 = r2_score(y_test, y_pred_log)

    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    })
    
    top_features = importances.sort_values(by='Importance', ascending=False).head(10)
    
    print(f"Model R²: {r2:.3f}")
    print(f"Mean Absolute Error (in millions of pounds): {mae:,.0f}")
    print("Top 10 Drivers of Value (Feature Importance):")
    for index, row in top_features.iterrows():
        print(f"  {row['Feature']:<15} : {row['Importance']:.4f}")
    print("-" * 40 + "\n")
    
    return rf, feature_cols

# rf_model, features_used = train_valuation_model(cb_df_clean)


def analyze_transfer_targets(df, model, min_value_ratio=1.2):
    """
    Applies the Value Model to find undervalued players.
    
    Args:
        df (pd.DataFrame): The player dataframe (cleaned).
        model: The trained Random Forest model.
        min_value_ratio (float): Minimum ROI to consider (default 1.2 = 20% undervalued).
        
    Returns:
        pd.DataFrame: A dataframe containing targets sorted by their Value Score.
    """
    targets = df.copy()
    
    excluded_cols = [
        'Transfer Value', 'Value_log', 'Wage', 'Wage_log', 'Last Trans. Fee', 'lastfee_log', 
        'Transfer Fees Received', 'totalfees_log', 'Name', 'Based', 'Division', 'Personality', 
        'Country', 'Position', 'Expires', 'Days Until Expiry', 'Begins', 'Age_at_Signing', 
        'Years_at_Club', 'Transfer Status', 'Transfer_Status_bool', 'WR', 'Rec', 'Inf'
    ]
    excluded_cols.extend([c for c in targets.columns if c.startswith('Country_')])
    
    feature_cols = [c for c in targets.columns 
                    if c not in excluded_cols 
                    and pd.api.types.is_numeric_dtype(targets[c])]
    
    predicted_log = model.predict(targets[feature_cols])
    targets['Intrinsic_Value'] = np.expm1(predicted_log)
    
    targets['Value_Ratio'] = targets['Intrinsic_Value'] / (targets['Transfer Value'] + 1)
    
    targets['Wage_Efficiency'] = targets['Intrinsic_Value'] / ((targets['Wage'] * 52) + 1)
    
    targets['Contract_Factor'] = np.where(targets['Days Until Expiry'] < 365, 1.25, 1.0)
    
    # Value Ratio 70% weight
    # Wage Efficiency 30% weight log-scaled
    # Contract leverage multiplier
    
    targets['Value_Score'] = (
        targets['Value_Ratio'] * np.log1p(targets['Wage_Efficiency']) * targets['Contract_Factor']
    )
    
    results = targets[targets['Value_Ratio'] > min_value_ratio].copy()
    
    output_cols = [
        'Name', 'Age', 'Position', 'Club', 'Based', 'Division', 'Nationality',
        'Transfer Value', 'Wage', 'Intrinsic_Value', 
        'Value_Ratio', 'Value_Score', 
        'Days Until Expiry'
    ]
    
    final_cols = [c for c in output_cols if c in results.columns]
    
    return results[final_cols].sort_values('Value_Score', ascending=False)

def analyze_market_inefficiencies(df_scored):
    """
    Aggregates player scores to find undervalued Leagues and Nations.
    
    Args:
        df_scored (pd.DataFrame): The output from analyze_transfer_targets() 
                                  (Make sure 'Based' and 'Division' are in the columns!)
                           
    Returns:
        pd.DataFrame: A report of the most undervalued markets.
    """
    
    market_stats = df_scored.groupby(['Based', 'Division']).agg({
        'Value_Ratio': ['count', 'median', 'mean'],
        'Value_Score': 'mean',
        'Wage': 'mean' 
    })
    
    market_stats.columns = [
        'Player_Count', 'Median_ROI', 'Mean_ROI', 
        'Avg_Value_Score', 'Avg_Wage'
    ]
    
    market_stats['Market_Attractiveness'] = (
        market_stats['Median_ROI'] * 10000 / (market_stats['Avg_Wage'] + 1)
    )
    
    # Sort by the most undervalued markets (Median ROI)
    return market_stats.sort_values('Median_ROI', ascending=False)




# scored_players = analyze_transfer_targets(cb_df, rf_model)

# best_leagues = analyze_market_inefficiencies(scored_players)

# print(best_leagues[['Player_Count', 'Median_ROI', 'Avg_Wage']].head(10))

