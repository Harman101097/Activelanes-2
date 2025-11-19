#!/usr/bin/env python3
"""
CAD/USD Volatility Containment Tester
=====================================
Interactive script to test how well a given volatility threshold
contains CAD/USD movements over different time periods with charting capabilities.

Double-click this file to run, or run from command line.
The script will automatically install missing dependencies.
"""

import sys
import subprocess
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Check for required packages and install if missing"""
    required_packages = {
        'requests': 'requests',
        'pandas': 'pandas', 
        'dateutil': 'python-dateutil',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            if module == 'dateutil':
                from dateutil.relativedelta import relativedelta
            elif module == 'matplotlib':
                import matplotlib.pyplot as plt
            elif module == 'seaborn':
                import seaborn as sns
            else:
                __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("="*60)
        print("INSTALLING REQUIRED PACKAGES")
        print("="*60)
        print("This script needs some additional packages to run.")
        print("Installing automatically...")
        print("")
        
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
                print(f"Please install manually: pip install {package}")
                return False
        
        print("\n✓ All packages installed successfully!")
        print("Restarting script...\n")
        
        # Restart the script
        os.execv(sys.executable, [sys.executable] + sys.argv)
    
    return True

# Check dependencies first
if __name__ == "__main__":
    if not check_and_install_dependencies():
        input("Press Enter to exit...")
        sys.exit(1)

# Now import everything
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from dateutil.relativedelta import relativedelta
import calendar

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    
    # Configure matplotlib for better display
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set matplotlib backend for compatibility
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
    
    PLOTTING_AVAILABLE = True
    print("📊 Charts enabled!")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("📊 Charts disabled - could not import plotting libraries")

class CADUSDVolatilityTester:
    def __init__(self):
        self.base_url = "https://www.bankofcanada.ca/valet"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
    
    def fetch_exchange_rate_data(self, series_name, start_date, end_date):
        """Fetch exchange rate data from Bank of Canada Valet API"""
        url = f"{self.base_url}/observations/{series_name}/json"
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        try:
            print("Fetching data from Bank of Canada...")
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if 'observations' in data:
                observations = data['observations']
                
                # Parse observations
                rows = []
                for obs in observations:
                    try:
                        date_val = obs.get('d')
                        # Handle different value structures
                        if series_name in obs:
                            if isinstance(obs[series_name], dict):
                                value_val = obs[series_name].get('v')
                            else:
                                value_val = obs[series_name]
                        else:
                            value_val = obs.get('v')
                        
                        if date_val and value_val is not None:
                            rows.append({'date': date_val, 'value': value_val})
                    except Exception as e:
                        continue
                
                if rows:
                    df = pd.DataFrame(rows)
                    df['date'] = pd.to_datetime(df['date'])
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    return df[['date', 'value']].dropna().sort_values('date')
                
            return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def is_business_day(self, date):
        """Check if date is a business day (Monday-Friday)"""
        return date.weekday() < 5
    
    def get_business_days_in_month(self, df, year, month):
        """Get all business days in a month with available data"""
        month_start = datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        month_end = datetime(year, month, last_day)
        
        # Filter data for this month
        month_data = df[(df['date'] >= month_start) & (df['date'] <= month_end)]
        
        # Get business days with data
        business_days = []
        for _, row in month_data.iterrows():
            if self.is_business_day(row['date']):
                business_days.append({
                    'date': row['date'],
                    'rate': row['value'],
                    'business_day_num': len(business_days) + 1
                })
        
        return business_days
    
    def calculate_rolling_movements(self, df, period_months):
        """Calculate rolling movements for specified period"""
        movements = []
        
        # Use appropriate data period based on containment period
        data_years = period_months  # 1 month = 1 year, 3 months = 3 years, etc.
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=data_years)
        
        print(f"Calculating {period_months}-month rolling movements using {data_years} years of data...")
        print(f"Data period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get all months in the period
        current_month = start_date.replace(day=1)
        months_processed = 0
        total_movements = 0
        
        while current_month < end_date - relativedelta(months=period_months):
            year = current_month.year
            month = current_month.month
            
            # Get business days for current month
            current_business_days = self.get_business_days_in_month(df, year, month)
            
            # Get business days for target month (current + period_months)
            target_month_date = current_month + relativedelta(months=period_months)
            target_business_days = self.get_business_days_in_month(df, target_month_date.year, target_month_date.month)
            
            if len(current_business_days) > 0 and len(target_business_days) > 0:
                # Compare same business day positions
                max_comparisons = min(len(current_business_days), len(target_business_days))
                
                for i in range(max_comparisons):
                    current_day = current_business_days[i]
                    target_day = target_business_days[i]
                    
                    # Calculate net movement (existing functionality)
                    movement = target_day['rate'] - current_day['rate']
                    percent_movement = (movement / current_day['rate']) * 100
                    
                    # Calculate max-min volatility during the period
                    period_start_date = current_day['date']
                    period_end_date = target_day['date']
                    
                    # Get all data points between start and end dates
                    period_data = df[(df['date'] >= period_start_date) & (df['date'] <= period_end_date)]
                    
                    max_min_volatility = 0
                    max_rate = current_day['rate']
                    min_rate = current_day['rate']
                    
                    if not period_data.empty and len(period_data) > 1:
                        max_rate = period_data['value'].max()
                        min_rate = period_data['value'].min()
                        # Calculate max-min volatility as percentage of starting rate
                        max_min_volatility = ((max_rate - min_rate) / current_day['rate']) * 100
                    
                    movements.append({
                        'start_date': current_day['date'].strftime('%Y-%m-%d'),
                        'end_date': target_day['date'].strftime('%Y-%m-%d'),
                        'business_day_position': i + 1,
                        'period_months': period_months,
                        'start_rate': current_day['rate'],
                        'end_rate': target_day['rate'],
                        'movement': movement,
                        'percent_movement': percent_movement,
                        'max_rate_in_period': max_rate,
                        'min_rate_in_period': min_rate,
                        'max_min_volatility': max_min_volatility
                    })
                    total_movements += 1
                
                months_processed += 1
                if months_processed % (data_years * 6) == 0:  # Progress update every 6 months per year of data
                    print(f"  Processed {months_processed} months, {total_movements} movements so far...")
            
            # Move to next month
            current_month = current_month + relativedelta(months=1)
        
        print(f"✓ Total {period_months}-month movements calculated: {total_movements}")
        return pd.DataFrame(movements)
    
    def analyze_containment(self, movements_df, volatility_threshold, period_months):
        """Analyze containment for given threshold and period"""
        if movements_df.empty:
            print("No movements to analyze")
            return None
        
        # Check containment
        movements_df['contained'] = movements_df['percent_movement'].abs() <= volatility_threshold
        
        contained_count = movements_df['contained'].sum()
        total_count = len(movements_df)
        containment_rate = (contained_count / total_count) * 100
        
        # Separate movements using CAD perspective:
        # Positive movement = CAD appreciating = rate going DOWN (negative percent_movement from rate perspective)
        # Negative movement = CAD depreciating = rate going UP (positive percent_movement from rate perspective)
        cad_appreciation_movements = movements_df[movements_df['percent_movement'] < 0]['percent_movement']  # Rate went down = CAD stronger
        cad_depreciation_movements = movements_df[movements_df['percent_movement'] > 0]['percent_movement']  # Rate went up = CAD weaker
        
        # Calculate statistics from CAD perspective
        results = {
            'period_months': period_months,
            'volatility_threshold': volatility_threshold,
            'total_movements': total_count,
            'contained_count': contained_count,
            'containment_rate': containment_rate,
            'failure_rate': 100 - containment_rate,
            'avg_cad_appreciation': abs(cad_appreciation_movements.mean()) if not cad_appreciation_movements.empty else 0,  # Convert to positive for CAD perspective
            'avg_cad_depreciation': cad_depreciation_movements.mean() if not cad_depreciation_movements.empty else 0,  # Keep positive as it represents CAD weakening
            'absolute_average': movements_df['percent_movement'].abs().mean(),
            'overall_average': movements_df['percent_movement'].mean(),
            'median_movement': movements_df['percent_movement'].median(),
            'std_deviation': movements_df['percent_movement'].std(),
            'min_movement': movements_df['percent_movement'].min(),
            'max_movement': movements_df['percent_movement'].max(),
            'cad_appreciation_count': len(cad_appreciation_movements),
            'cad_depreciation_count': len(cad_depreciation_movements)
        }
        
        return results, movements_df
    
    def display_results(self, results, movements_df):
        """Display comprehensive results"""
        period_name = f"{results['period_months']}-Month"
        
        print(f"\n" + "="*80)
        print(f"VOLATILITY CONTAINMENT ANALYSIS - {period_name.upper()} PERIOD")
        print("="*80)
        
        print(f"Testing Period: {period_name}")
        print(f"Volatility Threshold: ±{results['volatility_threshold']:.2f}%")
        print(f"Total Movements Analyzed: {results['total_movements']:,}")
        
        print(f"\n" + "="*60)
        print("CONTAINMENT RESULTS")
        print("="*60)
        print(f"Movements within ±{results['volatility_threshold']:.2f}%: {results['contained_count']:,}")
        print(f"Movements exceeding ±{results['volatility_threshold']:.2f}%: {results['total_movements'] - results['contained_count']:,}")
        print(f"")
        print(f"SUCCESS RATE: {results['containment_rate']:.2f}%")
        print(f"FAILURE RATE: {results['failure_rate']:.2f}%")
        
        print(f"\n" + "="*60)
        print("CAD MOVEMENT STATISTICS")
        print("="*60)
        print(f"Average CAD Appreciation: +{results['avg_cad_appreciation']:.4f}% (CAD strengthening)")
        print(f"Average CAD Depreciation: +{results['avg_cad_depreciation']:.4f}% (CAD weakening)")
        print(f"Absolute Average Movement: {results['absolute_average']:.4f}%")
        print(f"Overall Average Movement: {results['overall_average']:.4f}%")
        print(f"Median Movement: {results['median_movement']:.4f}%")
        print(f"Standard Deviation: {results['std_deviation']:.4f}%")
        
        print(f"\n" + "="*60)
        print("CAD DIRECTION ANALYSIS")
        print("="*60)
        print(f"CAD Appreciation periods: {results['cad_appreciation_count']:,} ({results['cad_appreciation_count']/results['total_movements']*100:.1f}%)")
        print(f"CAD Depreciation periods: {results['cad_depreciation_count']:,} ({results['cad_depreciation_count']/results['total_movements']*100:.1f}%)")
        
        print(f"\n" + "="*60)
        print("RANGE ANALYSIS")
        print("="*60)
        print(f"Largest CAD appreciation: {abs(results['min_movement']):.4f}% (rate decreased)")
        print(f"Largest CAD depreciation: +{results['max_movement']:.4f}% (rate increased)")
        print(f"Total range: {results['max_movement'] - results['min_movement']:.4f}%")
        
        # Show some extreme examples
        exceeded_movements = movements_df[~movements_df['contained']]
        if not exceeded_movements.empty:
            print(f"\nSample of movements that exceeded ±{results['volatility_threshold']:.2f}%:")
            sample_exceeded = exceeded_movements.sample(min(5, len(exceeded_movements)))
            for _, row in sample_exceeded.iterrows():
                if row['percent_movement'] < 0:
                    direction = f"CAD appreciated {abs(row['percent_movement']):.3f}%"
                else:
                    direction = f"CAD depreciated {row['percent_movement']:.3f}%"
                print(f"  {row['start_date']} → {row['end_date']}: {direction}")
    
    def analyze_directional_containment(self, movements_df, volatility_threshold):
        """Analyze containment separately for CAD appreciation and depreciation"""
        if movements_df.empty:
            print("No movements to analyze")
            return None
        
        # CAD Appreciation analysis (rate went down, negative percent_movement)
        cad_appreciation = movements_df[movements_df['percent_movement'] < 0].copy()
        if not cad_appreciation.empty:
            # Check if appreciation stayed within threshold (absolute value)
            cad_appreciation['contained'] = cad_appreciation['percent_movement'].abs() <= volatility_threshold
            appreciation_contained = cad_appreciation['contained'].sum()
            appreciation_total = len(cad_appreciation)
            appreciation_rate = (appreciation_contained / appreciation_total) * 100 if appreciation_total > 0 else 0
        else:
            appreciation_contained = 0
            appreciation_total = 0
            appreciation_rate = 0
        
        # CAD Depreciation analysis (rate went up, positive percent_movement)
        cad_depreciation = movements_df[movements_df['percent_movement'] > 0].copy()
        if not cad_depreciation.empty:
            # Check if depreciation stayed within threshold
            cad_depreciation['contained'] = cad_depreciation['percent_movement'] <= volatility_threshold
            depreciation_contained = cad_depreciation['contained'].sum()
            depreciation_total = len(cad_depreciation)
            depreciation_rate = (depreciation_contained / depreciation_total) * 100 if depreciation_total > 0 else 0
        else:
            depreciation_contained = 0
            depreciation_total = 0
            depreciation_rate = 0
        
        # Display results
        print(f"\n" + "="*70)
        print("DIRECTIONAL CONTAINMENT ANALYSIS")
        print("="*70)
        print(f"Analyzing CAD appreciation and depreciation separately...")
        print(f"Threshold: ±{volatility_threshold:.2f}%")
        
        print(f"\n📈 CAD APPRECIATION CONTAINMENT (CAD Strengthening):")
        if appreciation_total > 0:
            print(f"  Total appreciation periods: {appreciation_total:,}")
            print(f"  Contained within {volatility_threshold:.2f}%: {appreciation_contained:,}")
            print(f"  SUCCESS RATE: {appreciation_rate:.2f}%")
            print(f"  Average appreciation: {abs(cad_appreciation['percent_movement'].mean()):.4f}%")
        else:
            print(f"  No CAD appreciation periods found")
        
        print(f"\n📉 CAD DEPRECIATION CONTAINMENT (CAD Weakening):")
        if depreciation_total > 0:
            print(f"  Total depreciation periods: {depreciation_total:,}")
            print(f"  Contained within {volatility_threshold:.2f}%: {depreciation_contained:,}")
            print(f"  SUCCESS RATE: {depreciation_rate:.2f}%")
            print(f"  Average depreciation: {cad_depreciation['percent_movement'].mean():.4f}%")
        else:
            print(f"  No CAD depreciation periods found")
        
        print(f"\n📊 COMPARISON:")
        if appreciation_total > 0 and depreciation_total > 0:
            if appreciation_rate > depreciation_rate:
                diff = appreciation_rate - depreciation_rate
                print(f"  CAD appreciation is better contained (+{diff:.1f}% higher success rate)")
            elif depreciation_rate > appreciation_rate:
                diff = depreciation_rate - appreciation_rate
                print(f"  CAD depreciation is better contained (+{diff:.1f}% higher success rate)")
            else:
                print(f"  Both directions have similar containment rates")
        
        return {
            'appreciation_rate': appreciation_rate,
            'depreciation_rate': depreciation_rate,
            'appreciation_total': appreciation_total,
            'depreciation_total': depreciation_total,
            'appreciation_contained': appreciation_contained,
            'depreciation_contained': depreciation_contained
        }
    
    def analyze_max_min_volatility_containment(self, movements_df, volatility_threshold):
        """Analyze containment based on max-min volatility instead of net movement"""
        if movements_df.empty:
            print("No movements to analyze")
            return None
        
        # Check volatility containment (max-min range vs threshold)
        movements_df['volatility_contained'] = movements_df['max_min_volatility'] <= volatility_threshold
        
        contained_count = movements_df['volatility_contained'].sum()
        total_count = len(movements_df)
        containment_rate = (contained_count / total_count) * 100
        
        # Calculate volatility statistics
        volatility_stats = {
            'total_movements': total_count,
            'volatility_contained_count': contained_count,
            'volatility_containment_rate': containment_rate,
            'volatility_failure_rate': 100 - containment_rate,
            'avg_volatility': movements_df['max_min_volatility'].mean(),
            'median_volatility': movements_df['max_min_volatility'].median(),
            'std_volatility': movements_df['max_min_volatility'].std(),
            'min_volatility': movements_df['max_min_volatility'].min(),
            'max_volatility': movements_df['max_min_volatility'].max()
        }
        
        return volatility_stats, movements_df
    
    def display_max_min_volatility_results(self, volatility_stats, movements_df, volatility_threshold):
        """Display max-min volatility analysis results"""
        print(f"\n" + "="*80)
        print(f"MAX-MIN VOLATILITY CONTAINMENT ANALYSIS")
        print("="*80)
        print(f"Testing intra-period volatility range instead of net movement")
        print(f"Volatility Threshold: {volatility_threshold:.2f}% (max-min range)")
        print(f"Total Periods Analyzed: {volatility_stats['total_movements']:,}")
        
        print(f"\n" + "="*60)
        print("VOLATILITY CONTAINMENT RESULTS")
        print("="*60)
        print(f"Periods with volatility ≤ {volatility_threshold:.2f}%: {volatility_stats['volatility_contained_count']:,}")
        print(f"Periods with volatility > {volatility_threshold:.2f}%: {volatility_stats['total_movements'] - volatility_stats['volatility_contained_count']:,}")
        print(f"")
        print(f"VOLATILITY SUCCESS RATE: {volatility_stats['volatility_containment_rate']:.2f}%")
        print(f"VOLATILITY FAILURE RATE: {volatility_stats['volatility_failure_rate']:.2f}%")
        
        print(f"\n" + "="*60)
        print("VOLATILITY STATISTICS")
        print("="*60)
        print(f"Average Max-Min Volatility: {volatility_stats['avg_volatility']:.4f}%")
        print(f"Median Max-Min Volatility: {volatility_stats['median_volatility']:.4f}%")
        print(f"Standard Deviation: {volatility_stats['std_volatility']:.4f}%")
        print(f"Minimum Volatility: {volatility_stats['min_volatility']:.4f}%")
        print(f"Maximum Volatility: {volatility_stats['max_volatility']:.4f}%")
        
        # Compare with net movement results
        net_movement_avg = movements_df['percent_movement'].abs().mean()
        ratio = volatility_stats['avg_volatility'] / net_movement_avg if net_movement_avg != 0 else 0
        
        print(f"\n" + "="*60)
        print("COMPARISON WITH NET MOVEMENT")
        print("="*60)
        print(f"Average Net Movement (absolute): {net_movement_avg:.4f}%")
        print(f"Average Max-Min Volatility: {volatility_stats['avg_volatility']:.4f}%")
        print(f"Volatility/Net Movement Ratio: {ratio:.1f}x")
        print(f"")
        if ratio > 2:
            print(f"💡 Max-Min volatility is {ratio:.1f}x larger than net movements")
            print(f"   This suggests significant intra-period price swings")
        elif ratio > 1.5:
            print(f"💡 Moderate intra-period volatility ({ratio:.1f}x net movement)")
        else:
            print(f"💡 Low intra-period volatility ({ratio:.1f}x net movement)")
        
        # Show extreme volatility examples
        exceeded_volatility = movements_df[~movements_df['volatility_contained']]
        if not exceeded_volatility.empty:
            print(f"\nSample of periods with high volatility (>{volatility_threshold:.2f}%):")
            sample_exceeded = exceeded_volatility.nlargest(5, 'max_min_volatility')
            for _, row in sample_exceeded.iterrows():
                print(f"  {row['start_date']} → {row['end_date']}: Range {row['min_rate_in_period']:.4f}-{row['max_rate_in_period']:.4f} = {row['max_min_volatility']:.3f}% volatility")
    
    def analyze_directional_max_min_volatility(self, movements_df, volatility_threshold):
        """Analyze max-min volatility separately for CAD appreciation and depreciation dominant periods"""
        if movements_df.empty:
            print("No movements to analyze")
            return None
        
        # Classify periods by dominant direction (net movement)
        cad_appreciation_dominant = movements_df[movements_df['percent_movement'] < 0].copy()  # Net CAD strengthening
        cad_depreciation_dominant = movements_df[movements_df['percent_movement'] > 0].copy()  # Net CAD weakening
        
        print(f"\n" + "="*70)
        print("DIRECTIONAL MAX-MIN VOLATILITY ANALYSIS")
        print("="*70)
        print(f"Analyzing volatility in CAD appreciation vs depreciation dominant periods...")
        print(f"Volatility Threshold: {volatility_threshold:.2f}%")
        
        # CAD Appreciation Dominant Periods
        if not cad_appreciation_dominant.empty:
            cad_appreciation_dominant['volatility_contained'] = cad_appreciation_dominant['max_min_volatility'] <= volatility_threshold
            appreciation_vol_contained = cad_appreciation_dominant['volatility_contained'].sum()
            appreciation_vol_total = len(cad_appreciation_dominant)
            appreciation_vol_rate = (appreciation_vol_contained / appreciation_vol_total) * 100 if appreciation_vol_total > 0 else 0
            
            print(f"\n📈 CAD APPRECIATION DOMINANT PERIODS:")
            print(f"  Total periods: {appreciation_vol_total:,}")
            print(f"  Volatility contained: {appreciation_vol_contained:,}")
            print(f"  VOLATILITY SUCCESS RATE: {appreciation_vol_rate:.2f}%")
            print(f"  Average volatility: {cad_appreciation_dominant['max_min_volatility'].mean():.4f}%")
        else:
            appreciation_vol_rate = 0
            print(f"\n📈 CAD APPRECIATION DOMINANT PERIODS:")
            print(f"  No periods found")
        
        # CAD Depreciation Dominant Periods  
        if not cad_depreciation_dominant.empty:
            cad_depreciation_dominant['volatility_contained'] = cad_depreciation_dominant['max_min_volatility'] <= volatility_threshold
            depreciation_vol_contained = cad_depreciation_dominant['volatility_contained'].sum()
            depreciation_vol_total = len(cad_depreciation_dominant)
            depreciation_vol_rate = (depreciation_vol_contained / depreciation_vol_total) * 100 if depreciation_vol_total > 0 else 0
            
            print(f"\n📉 CAD DEPRECIATION DOMINANT PERIODS:")
            print(f"  Total periods: {depreciation_vol_total:,}")
            print(f"  Volatility contained: {depreciation_vol_contained:,}")
            print(f"  VOLATILITY SUCCESS RATE: {depreciation_vol_rate:.2f}%")
            print(f"  Average volatility: {cad_depreciation_dominant['max_min_volatility'].mean():.4f}%")
        else:
            depreciation_vol_rate = 0
            print(f"\n📉 CAD DEPRECIATION DOMINANT PERIODS:")
            print(f"  No periods found")
        
        # Comparison
        print(f"\n📊 VOLATILITY COMPARISON:")
        if appreciation_vol_rate > 0 and depreciation_vol_rate > 0:
            if appreciation_vol_rate > depreciation_vol_rate:
                diff = appreciation_vol_rate - depreciation_vol_rate
                print(f"  CAD appreciation periods have lower volatility (+{diff:.1f}% better containment)")
            elif depreciation_vol_rate > appreciation_vol_rate:
                diff = depreciation_vol_rate - appreciation_vol_rate
                print(f"  CAD depreciation periods have lower volatility (+{diff:.1f}% better containment)")
            else:
                print(f"  Both period types have similar volatility patterns")
        
        return {
            'appreciation_vol_rate': appreciation_vol_rate,
            'depreciation_vol_rate': depreciation_vol_rate,
            'appreciation_vol_total': len(cad_appreciation_dominant) if not cad_appreciation_dominant.empty else 0,
            'depreciation_vol_total': len(cad_depreciation_dominant) if not cad_depreciation_dominant.empty else 0
        }
    
    def ask_for_directional_analysis(self):
        """Ask if user wants directional analysis"""
        while True:
            directional = input("\nWould you like to analyze CAD appreciation and depreciation separately? (y/n): ").strip().lower()
            if directional in ['y', 'yes', 'n', 'no']:
                return directional in ['y', 'yes']
            print("Please enter 'y' or 'n'")
    
    def ask_for_max_min_volatility_analysis(self):
        """Ask if user wants max-min volatility analysis"""
        while True:
            volatility_analysis = input("\nWould you like to test volatility on absolute max-min range instead of net movement? (y/n): ").strip().lower()
            if volatility_analysis in ['y', 'yes', 'n', 'no']:
                return volatility_analysis in ['y', 'yes']
            print("Please enter 'y' or 'n'")
        """Ask if user wants directional analysis"""
        while True:
            directional = input("\nWould you like to analyze CAD appreciation and depreciation separately? (y/n): ").strip().lower()
            if directional in ['y', 'yes', 'n', 'no']:
                return directional in ['y', 'yes']
            print("Please enter 'y' or 'n'")
    
    def create_charts(self, movements_df, results, save_charts=False):
        """Create visualization charts for the analysis"""
        if not PLOTTING_AVAILABLE:
            print("Charts not available - plotting libraries not installed.")
            return
        
        try:
            period_name = f"{results['period_months']}-Month"
            threshold = results['volatility_threshold']
            
            # Create figure with subplots
            plt.figure(figsize=(16, 12))
            
            # Chart 1: Distribution of movements (CAD perspective)
            plt.subplot(2, 2, 1)
            n, bins, patches = plt.hist(movements_df['percent_movement'], bins=50, alpha=0.7, 
                                       color='skyblue', edgecolor='black')
            plt.axvline(-threshold, color='red', linestyle='--', linewidth=2, label=f'-{threshold:.2f}% threshold')
            plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'+{threshold:.2f}% threshold')
            plt.axvline(0, color='black', linestyle='-', alpha=0.5)
            
            # Color bars based on containment
            for i, (patch, bin_edge) in enumerate(zip(patches, bins[:-1])):
                if -threshold <= bin_edge <= threshold:
                    patch.set_facecolor('lightgreen')
                else:
                    patch.set_facecolor('lightcoral')
            
            plt.xlabel('Movement (%) - CAD Perspective')
            plt.ylabel('Frequency')
            plt.title('Distribution of CAD Movements\n(Negative = CAD Appreciation, Positive = CAD Depreciation)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Chart 2: Movements over time (CAD perspective)
            plt.subplot(2, 2, 2)
            movements_df['start_date'] = pd.to_datetime(movements_df['start_date'])
            movements_df_sorted = movements_df.sort_values('start_date')
            
            # Plot movements with CAD perspective colors
            contained = movements_df_sorted['contained']
            cad_appreciation = movements_df_sorted['percent_movement'] < 0
            cad_depreciation = movements_df_sorted['percent_movement'] > 0
            
            # Plot contained movements
            plt.scatter(movements_df_sorted[contained & cad_appreciation]['start_date'], 
                       movements_df_sorted[contained & cad_appreciation]['percent_movement'], 
                       c='green', alpha=0.6, s=3, label=f'CAD Appreciation (Contained)')
            plt.scatter(movements_df_sorted[contained & cad_depreciation]['start_date'], 
                       movements_df_sorted[contained & cad_depreciation]['percent_movement'], 
                       c='blue', alpha=0.6, s=3, label=f'CAD Depreciation (Contained)')
            
            # Plot exceeded movements
            plt.scatter(movements_df_sorted[~contained & cad_appreciation]['start_date'], 
                       movements_df_sorted[~contained & cad_appreciation]['percent_movement'], 
                       c='red', alpha=0.8, s=5, label=f'CAD Appreciation (Exceeded)')
            plt.scatter(movements_df_sorted[~contained & cad_depreciation]['start_date'], 
                       movements_df_sorted[~contained & cad_depreciation]['percent_movement'], 
                       c='orange', alpha=0.8, s=5, label=f'CAD Depreciation (Exceeded)')
            
            plt.axhline(threshold, color='red', linestyle='--', alpha=0.7)
            plt.axhline(-threshold, color='red', linestyle='--', alpha=0.7)
            plt.axhline(0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Date')
            plt.ylabel('Movement (%) - CAD Perspective')
            plt.title('CAD Movements Over Time')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.xticks(rotation=45)
            
            # Chart 3: Containment success/failure pie chart
            plt.subplot(2, 2, 3)
            contained_count = results['contained_count']
            failed_count = results['total_movements'] - contained_count
            
            sizes = [contained_count, failed_count]
            labels = [f'Contained\n({results["containment_rate"]:.1f}%)', 
                     f'Exceeded\n({results["failure_rate"]:.1f}%)']
            colors = ['lightgreen', 'lightcoral']
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title(f'Containment Results\n(±{threshold:.2f}% threshold)')
            
            # Chart 4: CAD directional statistics comparison
            plt.subplot(2, 2, 4)
            stats_labels = ['CAD\nAppreciation', 'CAD\nDepreciation', 'Absolute\nAvg', 'Std\nDev', f'Threshold\n(±{threshold:.2f}%)']
            stats_values = [
                results['avg_cad_appreciation'],
                results['avg_cad_depreciation'],
                results['absolute_average'],
                results['std_deviation'],
                threshold
            ]
            
            bars = plt.bar(range(len(stats_labels)), stats_values, 
                          color=['lightgreen', 'lightcoral', 'lightblue', 'lightyellow', 'red'])
            plt.xticks(range(len(stats_labels)), stats_labels)
            plt.ylabel('Percentage (%)')
            plt.title('CAD Movement Statistics Comparison')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, stats_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{value:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            plt.suptitle(f'CAD/USD Volatility Analysis - {period_name} Period (±{threshold:.2f}%)', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_charts:
                filename = f'cadusd_{results["period_months"]}month_analysis_{threshold:.2f}pct.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"✓ Chart saved as: {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating charts: {e}")
            print("Charts may not display properly on this system.")
    
    def get_save_preferences(self):
        """Ask user about saving preferences for both charts and data"""
        save_data = False
        show_charts = False
        save_charts = False
        
        # Ask about saving CSV data
        while True:
            save_data_input = input("\nSave detailed results to CSV file? (y/n): ").strip().lower()
            if save_data_input in ['y', 'yes', 'n', 'no']:
                save_data = save_data_input in ['y', 'yes']
                break
            print("Please enter 'y' or 'n'")
        
        # Ask about charts only if plotting is available
        if PLOTTING_AVAILABLE:
            while True:
                show_charts_input = input("Would you like to see charts? (y/n): ").strip().lower()
                if show_charts_input in ['y', 'yes', 'n', 'no']:
                    show_charts = show_charts_input in ['y', 'yes']
                    break
                print("Please enter 'y' or 'n'")
            
            if show_charts:
                while True:
                    save_charts_input = input("Save charts to file? (y/n): ").strip().lower()
                    if save_charts_input in ['y', 'yes', 'n', 'no']:
                        save_charts = save_charts_input in ['y', 'yes']
                        break
                    print("Please enter 'y' or 'n'")
        
        return save_data, show_charts, save_charts
    
    def ask_for_another_calculation(self):
        """Ask if user wants to run another test"""
        while True:
            another = input("\nWould you like to run another test? (y/n): ").strip().lower()
            if another in ['y', 'yes', 'n', 'no']:
                return another in ['y', 'yes']
            print("Please enter 'y' or 'n'")
    
    def get_user_inputs(self):
        """Get user inputs for period and threshold"""
        print("="*70)
        print("CAD/USD VOLATILITY CONTAINMENT TESTER")
        print("="*70)
        print("This tool tests how well a volatility threshold contains")
        print("CAD/USD movements over different time periods.")
        print("")
        print("Data periods used:")
        print("• 1 month period → 1 year of data")
        print("• 3 month period → 3 years of data") 
        print("• 6 month period → 6 years of data")
        print("• 12 month period → 12 years of data")
        print("")
        
        # Get containment period
        while True:
            try:
                print("Available containment periods:")
                print("1 = 1 month (uses 1 year of data)")
                print("3 = 3 months (uses 3 years of data)") 
                print("6 = 6 months (uses 6 years of data)")
                print("12 = 1 year (uses 12 years of data)")
                
                period_input = input("\nEnter containment period (1, 3, 6, or 12): ").strip()
                period_months = int(period_input)
                
                if period_months in [1, 3, 6, 12]:
                    break
                else:
                    print("Please enter 1, 3, 6, or 12 only.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get volatility threshold
        while True:
            try:
                threshold_input = input("\nEnter volatility threshold to test (e.g., 0.76 for ±0.76%): ").strip()
                volatility_threshold = float(threshold_input)
                
                if volatility_threshold > 0:
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
        
        return period_months, volatility_threshold
    
    def run_test(self):
        """Run the complete volatility test"""
        # Get user inputs
        period_months, volatility_threshold = self.get_user_inputs()
        
        # Get save preferences
        save_data, show_charts, save_charts = self.get_save_preferences()
        
        # Calculate data years needed based on period
        data_years = period_months  # 1 month = 1 year, 3 months = 3 years, etc.
        
        print(f"\nStarting analysis...")
        print(f"Containment Period: {period_months} month(s)")
        print(f"Threshold: ±{volatility_threshold:.2f}%")
        print(f"Data Period: {data_years} year(s)")
        
        # Fetch data with appropriate years
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=data_years)
        
        df = self.fetch_exchange_rate_data(
            'FXUSDCAD',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if df.empty:
            print("ERROR: Could not fetch data from Bank of Canada.")
            return None
        
        print(f"✓ Fetched {len(df):,} data points")
        print(f"✓ Data range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Calculate movements
        movements_df = self.calculate_rolling_movements(df, period_months)
        
        if movements_df.empty:
            print("ERROR: No movements calculated.")
            return None
        
        # Analyze containment
        results, movements_with_analysis = self.analyze_containment(movements_df, volatility_threshold, period_months)
        
        # Display results
        self.display_results(results, movements_with_analysis)
        
        # Ask for directional analysis
        if self.ask_for_directional_analysis():
            directional_results = self.analyze_directional_containment(movements_with_analysis, volatility_threshold)
        
        # Ask for max-min volatility analysis
        if self.ask_for_max_min_volatility_analysis():
            print(f"\nRunning max-min volatility analysis...")
            volatility_stats, movements_with_volatility = self.analyze_max_min_volatility_containment(movements_with_analysis, volatility_threshold)
            self.display_max_min_volatility_results(volatility_stats, movements_with_volatility, volatility_threshold)
            
            # Ask for directional max-min volatility analysis
            directional_vol_input = input("\nAnalyze max-min volatility by dominant direction (appreciation vs depreciation)? (y/n): ").strip().lower()
            if directional_vol_input in ['y', 'yes']:
                self.analyze_directional_max_min_volatility(movements_with_volatility, volatility_threshold)
        
        # Show charts if requested
        if show_charts:
            print(f"\nGenerating charts...")
            self.create_charts(movements_with_analysis, results, save_charts)
        
        # Save CSV data if requested
        if save_data:
            try:
                filename = f'cadusd_{period_months}month_test_{volatility_threshold:.2f}pct.csv'
                movements_with_analysis.to_csv(filename, index=False)
                print(f"\n✓ Detailed results saved to: {filename}")
            except Exception as e:
                print(f"\n❌ Could not save CSV file: {e}")
                print("This might be because:")
                print("  - The file is open in Excel or another program")
                print("  - No write permission in this folder")
                print("  - Antivirus software is blocking file creation")
        else:
            print(f"\n📋 Results not saved (as requested)")
        
        # Final summary
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Tested: ±{volatility_threshold:.2f}% over {period_months} month(s)")
        print(f"Data Period: {data_years} years ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')})")
        print(f"Success Rate: {results['containment_rate']:.2f}%")
        print(f"Total Movements: {results['total_movements']:,}")
        print(f"Absolute Average Movement: {results['absolute_average']:.4f}%")
        
        if results['containment_rate'] < 50:
            print(f"\n💡 Suggestion: Try a higher threshold (current {volatility_threshold:.2f}% only works {results['containment_rate']:.1f}% of the time)")
        elif results['containment_rate'] > 90:
            print(f"\n💡 Suggestion: You might be able to use a lower threshold (current {volatility_threshold:.2f}% works {results['containment_rate']:.1f}% of the time)")
        else:
            print(f"\n✓ Your {volatility_threshold:.2f}% threshold provides reasonable containment at {results['containment_rate']:.1f}% success rate")
        
        return results

def main():
    """Main function to run the volatility tester with loop for multiple tests"""
    try:
        print("="*70)
        print("CAD/USD VOLATILITY CONTAINMENT TESTER")
        print("="*70)
        print("Welcome! This tool uses appropriate historical data periods.")
        
        if PLOTTING_AVAILABLE:
            print("📊 Charts are enabled and ready!")
        else:
            print("📊 Charts disabled - plotting libraries not available")
        
        print("\nStarting application...")
        
        tester = CADUSDVolatilityTester()
        
        while True:
            try:
                results = tester.run_test()
                
                if results:
                    print(f"\n✅ Test completed successfully!")
                else:
                    print(f"\n❌ Test failed. Please check your internet connection and try again.")
                    break
                
                # Ask if user wants another calculation
                if not tester.ask_for_another_calculation():
                    print("\n🎉 Thank you for using the CAD/USD Volatility Tester!")
                    print("Application will close in 5 seconds...")
                    
                    # Keep window open briefly
                    import time
                    time.sleep(5)
                    break
                    
            except KeyboardInterrupt:
                print(f"\n\n⏹️  Test cancelled by user.")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Please check your internet connection and try again.")
                
                retry = input("\nWould you like to try again? (y/n): ").strip().lower()
                if retry not in ['y', 'yes']:
                    break
    
    except Exception as e:
        print(f"Application error: {e}")
        print("Press Enter to exit...")
        input()
    
    finally:
        # Ensure the window stays open if there's an error
        if 'pytest' not in sys.modules:  # Don't pause during testing
            try:
                import time
                time.sleep(2)
            except:
                pass

if __name__ == "__main__":
    main()