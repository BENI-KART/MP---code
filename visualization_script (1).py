"""
Advanced Visualization Script for Accident Severity Analysis
This script creates comprehensive visualizations for the case study
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


class AccidentVisualizer:
    """
    Create comprehensive visualizations for accident severity analysis
    """
    
    def __init__(self, df):
        self.df = df
        self.setup_style()
    
    def setup_style(self):
        """Set up visualization style"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def create_comprehensive_eda(self):
        """Create comprehensive EDA dashboard"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Severity Distribution
        ax1 = plt.subplot(3, 4, 1)
        severity_counts = self.df['Accident_Severity'].value_counts().sort_index()
        colors = ['#d62728', '#ff7f0e', '#2ca02c']
        severity_labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
        labels = [severity_labels.get(x, x) for x in severity_counts.index]
        
        wedges, texts, autotexts = ax1.pie(severity_counts, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax1.set_title('Accident Severity Distribution', fontweight='bold', fontsize=12)
        
        # 2. Time Series - Accidents by Hour
        ax2 = plt.subplot(3, 4, 2)
        if 'Hour' in self.df.columns:
            hourly = self.df['Hour'].value_counts().sort_index()
            ax2.plot(hourly.index, hourly.values, marker='o', linewidth=2, color='#1f77b4')
            ax2.fill_between(hourly.index, hourly.values, alpha=0.3)
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Number of Accidents')
            ax2.set_title('Accidents by Hour of Day', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 3. Weather Impact
        ax3 = plt.subplot(3, 4, 3)
        weather_severity = pd.crosstab(self.df['Weather_Conditions'], 
                                      self.df['Accident_Severity'], normalize='index') * 100
        weather_severity.plot(kind='bar', stacked=True, ax=ax3, color=colors)
        ax3.set_xlabel('Weather Conditions')
        ax3.set_ylabel('Percentage')
        ax3.set_title('Severity % by Weather', fontweight='bold')
        ax3.legend(title='Severity', labels=['Fatal', 'Serious', 'Slight'])
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Road Type Analysis
        ax4 = plt.subplot(3, 4, 4)
        road_counts = self.df['Road_Type'].value_counts()
        ax4.barh(road_counts.index, road_counts.values, color='#2ca02c')
        ax4.set_xlabel('Number of Accidents')
        ax4.set_title('Accidents by Road Type', fontweight='bold')
        
        # 5. Speed Limit vs Severity
        ax5 = plt.subplot(3, 4, 5)
        speed_severity = self.df.groupby(['Speed_limit', 'Accident_Severity']).size().unstack(fill_value=0)
        speed_severity.plot(kind='bar', ax=ax5, color=colors)
        ax5.set_xlabel('Speed Limit (mph)')
        ax5.set_ylabel('Count')
        ax5.set_title('Speed Limit vs Severity', fontweight='bold')
        ax5.legend(title='Severity', labels=['Fatal', 'Serious', 'Slight'])
        ax5.tick_params(axis='x', rotation=0)
        
        # 6. Light Conditions
        ax6 = plt.subplot(3, 4, 6)
        light_severity = pd.crosstab(self.df['Light_Conditions'], 
                                    self.df['Accident_Severity'])
        light_severity.plot(kind='bar', ax=ax6, color=colors)
        ax6.set_xlabel('Light Conditions')
        ax6.set_ylabel('Count')
        ax6.set_title('Light Conditions vs Severity', fontweight='bold')
        ax6.legend(title='Severity', labels=['Fatal', 'Serious', 'Slight'])
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. Urban vs Rural
        ax7 = plt.subplot(3, 4, 7)
        urban_rural = pd.crosstab(self.df['Urban_or_Rural_Area'], 
                                 self.df['Accident_Severity'], normalize='index') * 100
        urban_rural.plot(kind='bar', ax=ax7, color=colors)
        ax7.set_xlabel('Area Type')
        ax7.set_ylabel('Percentage')
        ax7.set_title('Urban vs Rural Severity %', fontweight='bold')
        ax7.legend(title='Severity', labels=['Fatal', 'Serious', 'Slight'])
        ax7.tick_params(axis='x', rotation=0)
        
        # 8. Number of Vehicles
        ax8 = plt.subplot(3, 4, 8)
        vehicle_counts = self.df['Number_of_Vehicles'].value_counts().sort_index()
        ax8.bar(vehicle_counts.index, vehicle_counts.values, color='#ff7f0e')
        ax8.set_xlabel('Number of Vehicles')
        ax8.set_ylabel('Count')
        ax8.set_title('Vehicles Involved Distribution', fontweight='bold')
        
        # 9. Casualties Distribution
        ax9 = plt.subplot(3, 4, 9)
        casualty_counts = self.df['Number_of_Casualties'].value_counts().sort_index()
        ax9.bar(casualty_counts.index, casualty_counts.values, color='#d62728')
        ax9.set_xlabel('Number of Casualties')
        ax9.set_ylabel('Count')
        ax9.set_title('Casualties Distribution', fontweight='bold')
        
        # 10. Day of Week Pattern
        ax10 = plt.subplot(3, 4, 10)
        if 'DayOfWeek' in self.df.columns:
            dow_counts = self.df['DayOfWeek'].value_counts().sort_index()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax10.plot(range(7), [dow_counts.get(i, 0) for i in range(7)], 
                     marker='o', linewidth=2, markersize=8, color='#9467bd')
            ax10.set_xticks(range(7))
            ax10.set_xticklabels(days)
            ax10.set_ylabel('Number of Accidents')
            ax10.set_title('Accidents by Day of Week', fontweight='bold')
            ax10.grid(True, alpha=0.3)
        
        # 11. Road Surface Conditions
        ax11 = plt.subplot(3, 4, 11)
        surface_severity = pd.crosstab(self.df['Road_Surface_Conditions'], 
                                      self.df['Accident_Severity'])
        surface_severity.plot(kind='bar', stacked=True, ax=ax11, color=colors)
        ax11.set_xlabel('Road Surface')
        ax11.set_ylabel('Count')
        ax11.set_title('Road Surface vs Severity', fontweight='bold')
        ax11.legend(title='Severity', labels=['Fatal', 'Serious', 'Slight'])
        ax11.tick_params(axis='x', rotation=45)
        
        # 12. Correlation Heatmap
        ax12 = plt.subplot(3, 4, 12)
        numeric_cols = ['Accident_Severity', 'Speed_limit', 'Number_of_Vehicles', 
                       'Number_of_Casualties']
        if all(col in self.df.columns for col in numeric_cols):
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax12, cbar_kws={'shrink': 0.8})
            ax12.set_title('Feature Correlations', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('comprehensive_eda_dashboard.png', dpi=300, bbox_inches='tight')
        print("Comprehensive EDA dashboard saved as 'comprehensive_eda_dashboard.png'")
        plt.close()
    
    def create_severity_heatmaps(self):
        """Create heatmaps showing severity patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Hour vs Day of Week
        if 'Hour' in self.df.columns and 'DayOfWeek' in self.df.columns:
            severity_pivot = self.df.groupby(['Hour', 'DayOfWeek'])['Accident_Severity'].mean().unstack()
            sns.heatmap(severity_pivot, cmap='RdYlGn_r', annot=False, ax=axes[0, 0], 
                       cbar_kws={'label': 'Avg Severity'})
            axes[0, 0].set_title('Accident Severity: Hour vs Day of Week', fontweight='bold')
            axes[0, 0].set_xlabel('Day of Week')
            axes[0, 0].set_ylabel('Hour of Day')
        
        # 2. Weather vs Light Conditions
        weather_light = pd.crosstab(self.df['Weather_Conditions'], 
                                    self.df['Light_Conditions'], 
                                    self.df['Accident_Severity'], aggfunc='mean')
        sns.heatmap(weather_light, cmap='RdYlGn_r', annot=True, fmt='.2f', ax=axes[0, 1],
                   cbar_kws={'label': 'Avg Severity'})
        axes[0, 1].set_title('Severity: Weather vs Light', fontweight='bold')
        
        # 3. Speed vs Road Type
        speed_road = pd.crosstab(self.df['Speed_limit'], 
                                self.df['Road_Type'], 
                                self.df['Accident_Severity'], aggfunc='mean')
        sns.heatmap(speed_road, cmap='RdYlGn_r', annot=True, fmt='.2f', ax=axes[1, 0],
                   cbar_kws={'label': 'Avg Severity'})
        axes[1, 0].set_title('Severity: Speed Limit vs Road Type', fontweight='bold')
        
        # 4. Vehicles vs Casualties
        if 'Number_of_Vehicles' in self.df.columns and 'Number_of_Casualties' in self.df.columns:
            veh_cas = pd.crosstab(self.df['Number_of_Vehicles'], 
                                 self.df['Number_of_Casualties'], 
                                 self.df['Accident_Severity'], aggfunc='mean')
            sns.heatmap(veh_cas, cmap='RdYlGn_r', annot=True, fmt='.2f', ax=axes[1, 1],
                       cbar_kws={'label': 'Avg Severity'})
            axes[1, 1].set_title('Severity: Vehicles vs Casualties', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('severity_heatmaps.png', dpi=300, bbox_inches='tight')
        print("Severity heatmaps saved as 'severity_heatmaps.png'")
        plt.close()
    
    def create_statistical_summary(self):
        """Create statistical summary visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Box plots for different severity levels
        severity_labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
        
        # 1. Speed Limit Distribution
        self.df['Severity_Label'] = self.df['Accident_Severity'].map(severity_labels)
        sns.boxplot(data=self.df, x='Severity_Label', y='Speed_limit', 
                   palette=['#d62728', '#ff7f0e', '#2ca02c'], ax=axes[0, 0])
        axes[0, 0].set_title('Speed Limit by Severity', fontweight='bold')
        axes[0, 0].set_xlabel('Severity')
        
        # 2. Number of Vehicles
        sns.violinplot(data=self.df, x='Severity_Label', y='Number_of_Vehicles', 
                      palette=['#d62728', '#ff7f0e', '#2ca02c'], ax=axes[0, 1])
        axes[0, 1].set_title('Vehicles by Severity', fontweight='bold')
        axes[0, 1].set_xlabel('Severity')
        
        # 3. Number of Casualties
        sns.violinplot(data=self.df, x='Severity_Label', y='Number_of_Casualties', 
                      palette=['#d62728', '#ff7f0e', '#2ca02c'], ax=axes[0, 2])
        axes[0, 2].set_title('Casualties by Severity', fontweight='bold')
        axes[0, 2].set_xlabel('Severity')
        
        # 4. Weather Conditions Count
        weather_counts = self.df['Weather_Conditions'].value_counts()
        axes[1, 0].pie(weather_counts, labels=weather_counts.index, autopct='%1.1f%%',
                      startangle=90)
        axes[1, 0].set_title('Weather Distribution', fontweight='bold')
        
        # 5. Road Type Count
        road_counts = self.df['Road_Type'].value_counts()
        axes[1, 1].pie(road_counts, labels=road_counts.index, autopct='%1.1f%%',
                      startangle=90)
        axes[1, 1].set_title('Road Type Distribution', fontweight='bold')
        
        # 6. Urban vs Rural
        area_counts = self.df['Urban_or_Rural_Area'].value_counts()
        axes[1, 2].pie(area_counts, labels=area_counts.index, autopct='%1.1f%%',
                      colors=['#1f77b4', '#ff7f0e'], startangle=90)
        axes[1, 2].set_title('Urban vs Rural Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('statistical_summary.png', dpi=300, bbox_inches='tight')
        print("Statistical summary saved as 'statistical_summary.png'")
        plt.close()
    
    def create_risk_analysis(self):
        """Create risk analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate risk scores
        # 1. Risk by Weather
        weather_risk = self.df.groupby('Weather_Conditions').agg({
            'Accident_Severity': ['mean', 'count']
        }).reset_index()
        weather_risk.columns = ['Weather', 'Avg_Severity', 'Count']
        weather_risk['Risk_Score'] = weather_risk['Avg_Severity'] * np.log(weather_risk['Count'])
        weather_risk = weather_risk.sort_values('Risk_Score', ascending=False)
        
        axes[0, 0].barh(weather_risk['Weather'], weather_risk['Risk_Score'], 
                       color='#d62728')
        axes[0, 0].set_xlabel('Risk Score')
        axes[0, 0].set_title('Weather Condition Risk Analysis', fontweight='bold')
        axes[0, 0].invert_yaxis()
        
        # 2. Risk by Road Type
        road_risk = self.df.groupby('Road_Type').agg({
            'Accident_Severity': ['mean', 'count']
        }).reset_index()
        road_risk.columns = ['Road_Type', 'Avg_Severity', 'Count']
        road_risk['Risk_Score'] = road_risk['Avg_Severity'] * np.log(road_risk['Count'])
        road_risk = road_risk.sort_values('Risk_Score', ascending=False)
        
        axes[0, 1].barh(road_risk['Road_Type'], road_risk['Risk_Score'], 
                       color='#ff7f0e')
        axes[0, 1].set_xlabel('Risk Score')
        axes[0, 1].set_title('Road Type Risk Analysis', fontweight='bold')
        axes[0, 1].invert_yaxis()
        
        # 3. Risk by Speed Limit
        speed_stats = self.df.groupby('Speed_limit').agg({
            'Accident_Severity': ['mean', 'count']
        }).reset_index()
        speed_stats.columns = ['Speed_Limit', 'Avg_Severity', 'Count']
        
        axes[1, 0].scatter(speed_stats['Speed_Limit'], speed_stats['Avg_Severity'], 
                          s=speed_stats['Count']*2, alpha=0.6, c=speed_stats['Avg_Severity'],
                          cmap='RdYlGn_r')
        axes[1, 0].set_xlabel('Speed Limit (mph)')
        axes[1, 0].set_ylabel('Average Severity')
        axes[1, 0].set_title('Speed Limit vs Severity (bubble size = count)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Time-based Risk Pattern
        if 'Hour' in self.df.columns:
            hourly_risk = self.df.groupby('Hour')['Accident_Severity'].agg(['mean', 'count'])
            
            ax_twin = axes[1, 1].twinx()
            axes[1, 1].bar(hourly_risk.index, hourly_risk['count'], alpha=0.3, 
                          color='gray', label='Count')
            ax_twin.plot(hourly_risk.index, hourly_risk['mean'], color='red', 
                        linewidth=2, marker='o', label='Avg Severity')
            
            axes[1, 1].set_xlabel('Hour of Day')
            axes[1, 1].set_ylabel('Accident Count', color='gray')
            ax_twin.set_ylabel('Average Severity', color='red')
            axes[1, 1].set_title('Hourly Accident Pattern & Severity', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('risk_analysis.png', dpi=300, bbox_inches='tight')
        print("Risk analysis saved as 'risk_analysis.png'")
        plt.close()
    
    def create_all_visualizations(self):
        """Generate all visualizations"""
        print("\n=== CREATING VISUALIZATIONS ===")
        self.create_comprehensive_eda()
        self.create_severity_heatmaps()
        self.create_statistical_summary()
        self.create_risk_analysis()
        print("\nAll visualizations created successfully!")


def main():
    """Main execution"""
    # Load sample data (generated from main script)
    print("Loading data...")
    
    # Check if sample data exists
    try:
        df = pd.read_csv('sample_accident_data.csv')
        print(f"Loaded {len(df)} accident records")
    except FileNotFoundError:
        print("Sample data not found. Generating new data...")
        from accident_severity_prediction import generate_sample_data
        df = generate_sample_data(10000)
        df.to_csv('sample_accident_data.csv', index=False)
    
    # Create visualizer
    visualizer = AccidentVisualizer(df)
    
    # Generate all visualizations
    visualizer.create_all_visualizations()
    
    print("\n✓ Visualization pipeline completed!")
    print("\nGenerated files:")
    print("  - comprehensive_eda_dashboard.png")
    print("  - severity_heatmaps.png")
    print("  - statistical_summary.png")
    print("  - risk_analysis.png")


if __name__ == "__main__":
    main()
