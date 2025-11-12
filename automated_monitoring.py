"""
Automated Data Quality Monitoring Scheduler
Continuous monitoring and alerting system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

class DataQualityMonitor:
    """
    Automated data quality monitoring system with alerting
    """
    
    def __init__(self, thresholds=None):
        """Initialize monitor with quality thresholds"""
        self.thresholds = thresholds or {
            'completeness_min': 0.95,
            'validity_min': 0.90,
            'consistency_min': 0.95,
            'accuracy_min': 0.90,
            'overall_min': 0.85
        }
        
        self.alerts = []
        self.history = []
        
    def load_data(self, filepath):
        """Load dataset for monitoring"""
        try:
            self.data = pd.read_csv(filepath)
            self.filepath = filepath
            return True
        except Exception as e:
            self.add_alert('ERROR', f'Failed to load data: {str(e)}')
            return False
    
    def check_completeness(self):
        """Check data completeness"""
        total_cells = self.data.shape[0] * self.data.shape[1]
        missing_cells = self.data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)
        
        if completeness < self.thresholds['completeness_min']:
            self.add_alert('WARNING', 
                         f'Completeness {completeness:.2%} below threshold {self.thresholds["completeness_min"]:.2%}')
        
        return completeness
    
    def check_freshness(self, date_column, max_age_days=7):
        """Check data freshness"""
        if date_column not in self.data.columns:
            return None
        
        try:
            dates = pd.to_datetime(self.data[date_column], errors='coerce')
            latest_date = dates.max()
            age_days = (pd.Timestamp.now() - latest_date).days
            
            if age_days > max_age_days:
                self.add_alert('WARNING', 
                             f'Data is {age_days} days old (threshold: {max_age_days} days)')
            
            return age_days
        except:
            return None
    
    def check_volume(self, min_records=100):
        """Check if sufficient data volume"""
        record_count = len(self.data)
        
        if record_count < min_records:
            self.add_alert('CRITICAL', 
                         f'Only {record_count} records found (minimum: {min_records})')
        
        return record_count
    
    def check_duplicates(self, max_duplicate_pct=0.02):
        """Check for duplicate records"""
        duplicate_count = self.data.duplicated().sum()
        duplicate_pct = duplicate_count / len(self.data)
        
        if duplicate_pct > max_duplicate_pct:
            self.add_alert('WARNING', 
                         f'Duplicate rate {duplicate_pct:.2%} exceeds threshold {max_duplicate_pct:.2%}')
        
        return duplicate_pct
    
    def check_schema(self, expected_columns):
        """Validate data schema"""
        actual_columns = set(self.data.columns)
        expected = set(expected_columns)
        
        missing = expected - actual_columns
        extra = actual_columns - expected
        
        if missing:
            self.add_alert('ERROR', f'Missing columns: {missing}')
        
        if extra:
            self.add_alert('INFO', f'Extra columns found: {extra}')
        
        return len(missing) == 0
    
    def add_alert(self, level, message):
        """Add alert to queue"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.alerts.append(alert)
    
    def run_checks(self, date_column=None):
        """Run all monitoring checks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'filepath': self.filepath,
            'checks': {}
        }
        
        # Run checks
        results['checks']['completeness'] = self.check_completeness()
        results['checks']['volume'] = self.check_volume()
        results['checks']['duplicates'] = self.check_duplicates()
        
        if date_column:
            results['checks']['freshness_days'] = self.check_freshness(date_column)
        
        results['alerts'] = self.alerts
        
        # Save to history
        self.history.append(results)
        
        return results
    
    def generate_alert_report(self, output_file='data_quality_reports/alerts.txt'):
        """Generate alert report"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DATA QUALITY ALERTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            if not self.alerts:
                f.write("✓ No alerts - all checks passed!\n")
            else:
                for alert in self.alerts:
                    f.write(f"[{alert['level']}] {alert['timestamp']}\n")
                    f.write(f"  {alert['message']}\n\n")
    
    def save_history(self, output_file='data_quality_reports/monitoring_history.json'):
        """Save monitoring history"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, default=str)


def schedule_monitoring(data_path, schedule_type='daily'):
    """
    Schedule automated monitoring
    
    Parameters:
    - data_path: Path to data file
    - schedule_type: 'daily', 'weekly', or 'monthly'
    """
    
    print("=" * 80)
    print("AUTOMATED DATA QUALITY MONITORING")
    print("=" * 80)
    
    monitor = DataQualityMonitor()
    
    if monitor.load_data(data_path):
        print(f"✓ Loaded data: {data_path}")
        print(f"  Records: {len(monitor.data)}")
        print(f"  Columns: {len(monitor.data.columns)}")
        
        # Run checks
        results = monitor.run_checks(date_column='Date (DD/MM/YYYY)')
        
        print(f"\n✓ Monitoring checks completed")
        print(f"  Completeness: {results['checks']['completeness']:.2%}")
        print(f"  Volume: {results['checks']['volume']} records")
        print(f"  Duplicates: {results['checks']['duplicates']:.2%}")
        
        if 'freshness_days' in results['checks'] and results['checks']['freshness_days']:
            print(f"  Freshness: {results['checks']['freshness_days']} days old")
        
        print(f"\n  Alerts: {len(monitor.alerts)}")
        
        # Generate reports
        monitor.generate_alert_report()
        monitor.save_history()
        
        print(f"\n✓ Reports saved to data_quality_reports/")
        
        return monitor
    else:
        print("✗ Failed to load data")
        return None


if __name__ == "__main__":
    # Run monitoring
    monitor = schedule_monitoring(
        data_path=r'c:\Users\Millpark\Downloads\River water parameters.csv',
        schedule_type='daily'
    )
    
    if monitor:
        print("\n" + "=" * 80)
        print("MONITORING SUMMARY")
        print("=" * 80)
        
        if monitor.alerts:
            print(f"\n⚠ {len(monitor.alerts)} alert(s) generated:")
            for alert in monitor.alerts:
                print(f"  [{alert['level']}] {alert['message']}")
        else:
            print("\n✓ All quality checks passed - no alerts!")
        
        print("\n" + "=" * 80)
