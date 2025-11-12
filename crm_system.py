"""
Customer Relationship Management (CRM) System
Advanced CRM with customer tracking, interaction logging, and analytics
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class CRMDatabase:
    """Database manager for CRM system"""
    
    def __init__(self, db_name: str = 'crm_database.db'):
        """Initialize database connection"""
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        print(f"✓ Connected to database: {self.db_name}")
    
    def create_tables(self):
        """Create all necessary tables"""
        
        # Customers table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                company TEXT,
                industry TEXT,
                customer_type TEXT,
                status TEXT DEFAULT 'Active',
                lifetime_value REAL DEFAULT 0.0,
                created_date TEXT,
                last_contact_date TEXT,
                notes TEXT
            )
        ''')
        
        # Interactions table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER,
                interaction_type TEXT,
                interaction_date TEXT,
                subject TEXT,
                description TEXT,
                outcome TEXT,
                next_action TEXT,
                follow_up_date TEXT,
                duration_minutes INTEGER,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
        ''')
        
        # Sales/Opportunities table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS opportunities (
                opportunity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER,
                opportunity_name TEXT,
                value REAL,
                stage TEXT,
                probability INTEGER,
                expected_close_date TEXT,
                created_date TEXT,
                closed_date TEXT,
                status TEXT DEFAULT 'Open',
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
        ''')
        
        # Products/Services table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT UNIQUE NOT NULL,
                category TEXT,
                price REAL,
                description TEXT,
                active INTEGER DEFAULT 1
            )
        ''')
        
        # Transactions table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER,
                product_id INTEGER,
                transaction_date TEXT,
                quantity INTEGER,
                unit_price REAL,
                total_amount REAL,
                payment_method TEXT,
                status TEXT DEFAULT 'Completed',
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
                FOREIGN KEY (product_id) REFERENCES products (product_id)
            )
        ''')
        
        # Tasks table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER,
                task_title TEXT,
                description TEXT,
                assigned_to TEXT,
                priority TEXT,
                due_date TEXT,
                status TEXT DEFAULT 'Pending',
                created_date TEXT,
                completed_date TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
        ''')
        
        self.conn.commit()
        print("✓ Database tables created successfully")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")


class CustomerManager:
    """Manage customer operations"""
    
    def __init__(self, db: CRMDatabase):
        self.db = db
    
    def add_customer(self, first_name: str, last_name: str, email: str, 
                    phone: str = None, company: str = None, industry: str = None,
                    customer_type: str = 'Prospect', notes: str = None) -> int:
        """Add a new customer"""
        try:
            created_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            self.db.cursor.execute('''
                INSERT INTO customers (first_name, last_name, email, phone, company, 
                                     industry, customer_type, created_date, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (first_name, last_name, email, phone, company, industry, 
                  customer_type, created_date, notes))
            
            self.db.conn.commit()
            customer_id = self.db.cursor.lastrowid
            print(f"✓ Customer added: {first_name} {last_name} (ID: {customer_id})")
            return customer_id
            
        except sqlite3.IntegrityError:
            print(f"✗ Error: Email {email} already exists")
            return -1
    
    def update_customer(self, customer_id: int, **kwargs):
        """Update customer information"""
        valid_fields = ['first_name', 'last_name', 'email', 'phone', 'company', 
                       'industry', 'customer_type', 'status', 'notes']
        
        updates = []
        values = []
        
        for field, value in kwargs.items():
            if field in valid_fields:
                updates.append(f"{field} = ?")
                values.append(value)
        
        if updates:
            values.append(customer_id)
            query = f"UPDATE customers SET {', '.join(updates)} WHERE customer_id = ?"
            self.db.cursor.execute(query, values)
            self.db.conn.commit()
            print(f"✓ Customer {customer_id} updated successfully")
        else:
            print("✗ No valid fields to update")
    
    def get_customer(self, customer_id: int) -> Optional[Dict]:
        """Get customer details"""
        self.db.cursor.execute('SELECT * FROM customers WHERE customer_id = ?', (customer_id,))
        row = self.db.cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in self.db.cursor.description]
            return dict(zip(columns, row))
        return None
    
    def search_customers(self, search_term: str = None, customer_type: str = None, 
                        status: str = None) -> pd.DataFrame:
        """Search customers with filters"""
        query = "SELECT * FROM customers WHERE 1=1"
        params = []
        
        if search_term:
            query += " AND (first_name LIKE ? OR last_name LIKE ? OR email LIKE ? OR company LIKE ?)"
            search_pattern = f'%{search_term}%'
            params.extend([search_pattern] * 4)
        
        if customer_type:
            query += " AND customer_type = ?"
            params.append(customer_type)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        df = pd.read_sql_query(query, self.db.conn, params=params)
        return df
    
    def delete_customer(self, customer_id: int):
        """Delete a customer (soft delete by marking inactive)"""
        self.db.cursor.execute('UPDATE customers SET status = ? WHERE customer_id = ?', 
                              ('Inactive', customer_id))
        self.db.conn.commit()
        print(f"✓ Customer {customer_id} marked as Inactive")


class InteractionManager:
    """Manage customer interactions"""
    
    def __init__(self, db: CRMDatabase):
        self.db = db
    
    def log_interaction(self, customer_id: int, interaction_type: str, 
                       subject: str, description: str = None, outcome: str = None,
                       next_action: str = None, follow_up_date: str = None,
                       duration_minutes: int = None) -> int:
        """Log a customer interaction"""
        interaction_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.db.cursor.execute('''
            INSERT INTO interactions (customer_id, interaction_type, interaction_date,
                                    subject, description, outcome, next_action, 
                                    follow_up_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (customer_id, interaction_type, interaction_date, subject, description,
              outcome, next_action, follow_up_date, duration_minutes))
        
        # Update last contact date for customer
        self.db.cursor.execute('''
            UPDATE customers SET last_contact_date = ? WHERE customer_id = ?
        ''', (interaction_date, customer_id))
        
        self.db.conn.commit()
        interaction_id = self.db.cursor.lastrowid
        print(f"✓ Interaction logged (ID: {interaction_id})")
        return interaction_id
    
    def get_customer_interactions(self, customer_id: int) -> pd.DataFrame:
        """Get all interactions for a customer"""
        query = '''
            SELECT i.*, c.first_name, c.last_name
            FROM interactions i
            JOIN customers c ON i.customer_id = c.customer_id
            WHERE i.customer_id = ?
            ORDER BY i.interaction_date DESC
        '''
        df = pd.read_sql_query(query, self.db.conn, params=(customer_id,))
        return df
    
    def get_upcoming_followups(self, days_ahead: int = 7) -> pd.DataFrame:
        """Get upcoming follow-ups"""
        today = datetime.now().strftime('%Y-%m-%d')
        future_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        query = '''
            SELECT i.*, c.first_name, c.last_name, c.email, c.phone
            FROM interactions i
            JOIN customers c ON i.customer_id = c.customer_id
            WHERE i.follow_up_date BETWEEN ? AND ?
            ORDER BY i.follow_up_date
        '''
        df = pd.read_sql_query(query, self.db.conn, params=(today, future_date))
        return df


class OpportunityManager:
    """Manage sales opportunities"""
    
    def __init__(self, db: CRMDatabase):
        self.db = db
    
    def create_opportunity(self, customer_id: int, opportunity_name: str,
                          value: float, stage: str = 'Prospecting',
                          probability: int = 10, expected_close_date: str = None) -> int:
        """Create a new sales opportunity"""
        created_date = datetime.now().strftime('%Y-%m-%d')
        
        self.db.cursor.execute('''
            INSERT INTO opportunities (customer_id, opportunity_name, value, stage,
                                     probability, expected_close_date, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (customer_id, opportunity_name, value, stage, probability, 
              expected_close_date, created_date))
        
        self.db.conn.commit()
        opportunity_id = self.db.cursor.lastrowid
        print(f"✓ Opportunity created: {opportunity_name} (ID: {opportunity_id})")
        return opportunity_id
    
    def update_opportunity_stage(self, opportunity_id: int, stage: str, 
                                 probability: int = None):
        """Update opportunity stage"""
        if probability is None:
            stage_probabilities = {
                'Prospecting': 10,
                'Qualification': 25,
                'Proposal': 50,
                'Negotiation': 75,
                'Closed Won': 100,
                'Closed Lost': 0
            }
            probability = stage_probabilities.get(stage, 10)
        
        # Check if won or lost
        status = 'Closed' if stage in ['Closed Won', 'Closed Lost'] else 'Open'
        closed_date = datetime.now().strftime('%Y-%m-%d') if status == 'Closed' else None
        
        self.db.cursor.execute('''
            UPDATE opportunities 
            SET stage = ?, probability = ?, status = ?, closed_date = ?
            WHERE opportunity_id = ?
        ''', (stage, probability, status, closed_date, opportunity_id))
        
        self.db.conn.commit()
        print(f"✓ Opportunity {opportunity_id} updated to stage: {stage}")
    
    def get_pipeline(self) -> pd.DataFrame:
        """Get sales pipeline"""
        query = '''
            SELECT o.*, c.first_name, c.last_name, c.company
            FROM opportunities o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.status = 'Open'
            ORDER BY o.expected_close_date
        '''
        df = pd.read_sql_query(query, self.db.conn)
        return df
    
    def get_forecast(self) -> Dict:
        """Get sales forecast based on probabilities"""
        df = self.get_pipeline()
        
        if len(df) == 0:
            return {
                'total_pipeline': 0,
                'weighted_forecast': 0,
                'best_case': 0,
                'worst_case': 0,
                'opportunities_count': 0
            }
        
        df['weighted_value'] = df['value'] * (df['probability'] / 100)
        
        return {
            'total_pipeline': df['value'].sum(),
            'weighted_forecast': df['weighted_value'].sum(),
            'best_case': df[df['probability'] >= 75]['value'].sum(),
            'worst_case': df[df['probability'] >= 90]['value'].sum(),
            'opportunities_count': len(df)
        }


class ProductManager:
    """Manage products and services"""
    
    def __init__(self, db: CRMDatabase):
        self.db = db
    
    def add_product(self, product_name: str, category: str, price: float,
                   description: str = None) -> int:
        """Add a new product"""
        try:
            self.db.cursor.execute('''
                INSERT INTO products (product_name, category, price, description)
                VALUES (?, ?, ?, ?)
            ''', (product_name, category, price, description))
            
            self.db.conn.commit()
            product_id = self.db.cursor.lastrowid
            print(f"✓ Product added: {product_name} (ID: {product_id})")
            return product_id
            
        except sqlite3.IntegrityError:
            print(f"✗ Error: Product {product_name} already exists")
            return -1
    
    def get_products(self, active_only: bool = True) -> pd.DataFrame:
        """Get all products"""
        query = "SELECT * FROM products"
        if active_only:
            query += " WHERE active = 1"
        df = pd.read_sql_query(query, self.db.conn)
        return df


class TransactionManager:
    """Manage customer transactions"""
    
    def __init__(self, db: CRMDatabase):
        self.db = db
    
    def record_transaction(self, customer_id: int, product_id: int,
                          quantity: int, unit_price: float,
                          payment_method: str = 'Credit Card') -> int:
        """Record a customer transaction"""
        transaction_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_amount = quantity * unit_price
        
        self.db.cursor.execute('''
            INSERT INTO transactions (customer_id, product_id, transaction_date,
                                    quantity, unit_price, total_amount, payment_method)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (customer_id, product_id, transaction_date, quantity, unit_price,
              total_amount, payment_method))
        
        # Update customer lifetime value
        self.db.cursor.execute('''
            UPDATE customers 
            SET lifetime_value = lifetime_value + ?
            WHERE customer_id = ?
        ''', (total_amount, customer_id))
        
        self.db.conn.commit()
        transaction_id = self.db.cursor.lastrowid
        print(f"✓ Transaction recorded: ${total_amount:.2f} (ID: {transaction_id})")
        return transaction_id
    
    def get_customer_transactions(self, customer_id: int) -> pd.DataFrame:
        """Get all transactions for a customer"""
        query = '''
            SELECT t.*, p.product_name, p.category
            FROM transactions t
            JOIN products p ON t.product_id = p.product_id
            WHERE t.customer_id = ?
            ORDER BY t.transaction_date DESC
        '''
        df = pd.read_sql_query(query, self.db.conn, params=(customer_id,))
        return df


class CRMAnalytics:
    """Analytics and reporting for CRM"""
    
    def __init__(self, db: CRMDatabase):
        self.db = db
    
    def customer_summary(self) -> Dict:
        """Get overall customer summary statistics"""
        query = '''
            SELECT 
                COUNT(*) as total_customers,
                SUM(CASE WHEN status = 'Active' THEN 1 ELSE 0 END) as active_customers,
                SUM(CASE WHEN customer_type = 'Client' THEN 1 ELSE 0 END) as clients,
                SUM(CASE WHEN customer_type = 'Prospect' THEN 1 ELSE 0 END) as prospects,
                SUM(lifetime_value) as total_lifetime_value,
                AVG(lifetime_value) as avg_lifetime_value
            FROM customers
        '''
        self.db.cursor.execute(query)
        row = self.db.cursor.fetchone()
        columns = [desc[0] for desc in self.db.cursor.description]
        return dict(zip(columns, row))
    
    def sales_summary(self) -> Dict:
        """Get sales summary statistics"""
        query = '''
            SELECT 
                COUNT(*) as total_transactions,
                SUM(total_amount) as total_revenue,
                AVG(total_amount) as avg_transaction_value,
                MAX(total_amount) as largest_transaction
            FROM transactions
        '''
        self.db.cursor.execute(query)
        row = self.db.cursor.fetchone()
        columns = [desc[0] for desc in self.db.cursor.description]
        return dict(zip(columns, row))
    
    def top_customers(self, limit: int = 10) -> pd.DataFrame:
        """Get top customers by lifetime value"""
        query = '''
            SELECT customer_id, first_name, last_name, email, company, 
                   lifetime_value, customer_type
            FROM customers
            WHERE status = 'Active'
            ORDER BY lifetime_value DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, self.db.conn, params=(limit,))
        return df
    
    def interaction_summary(self) -> pd.DataFrame:
        """Summary of interactions by type"""
        query = '''
            SELECT interaction_type, COUNT(*) as count,
                   AVG(duration_minutes) as avg_duration
            FROM interactions
            WHERE interaction_date >= date('now', '-30 days')
            GROUP BY interaction_type
            ORDER BY count DESC
        '''
        df = pd.read_sql_query(query, self.db.conn)
        return df
    
    def revenue_by_product(self) -> pd.DataFrame:
        """Revenue breakdown by product"""
        query = '''
            SELECT p.product_name, p.category,
                   COUNT(t.transaction_id) as transactions,
                   SUM(t.total_amount) as total_revenue
            FROM transactions t
            JOIN products p ON t.product_id = p.product_id
            GROUP BY p.product_id
            ORDER BY total_revenue DESC
        '''
        df = pd.read_sql_query(query, self.db.conn)
        return df
    
    def create_dashboard(self, output_file: str = 'crm_dashboard.png'):
        """Create visual dashboard"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Customer status distribution
        ax1 = fig.add_subplot(gs[0, 0])
        customer_df = pd.read_sql_query('SELECT customer_type FROM customers', self.db.conn)
        customer_df['customer_type'].value_counts().plot(kind='pie', ax=ax1, autopct='%1.1f%%')
        ax1.set_title('Customer Type Distribution', fontweight='bold', fontsize=12)
        ax1.set_ylabel('')
        
        # Revenue trend (last 30 days)
        ax2 = fig.add_subplot(gs[0, 1:])
        revenue_query = '''
            SELECT DATE(transaction_date) as date, SUM(total_amount) as revenue
            FROM transactions
            WHERE transaction_date >= date('now', '-30 days')
            GROUP BY DATE(transaction_date)
            ORDER BY date
        '''
        revenue_df = pd.read_sql_query(revenue_query, self.db.conn)
        if len(revenue_df) > 0:
            ax2.plot(revenue_df['date'], revenue_df['revenue'], marker='o', linewidth=2)
            ax2.set_title('Revenue Trend (Last 30 Days)', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Revenue ($)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # Top customers
        ax3 = fig.add_subplot(gs[1, :])
        top_cust = self.top_customers(limit=10)
        if len(top_cust) > 0:
            top_cust['name'] = top_cust['first_name'] + ' ' + top_cust['last_name']
            ax3.barh(top_cust['name'], top_cust['lifetime_value'])
            ax3.set_title('Top 10 Customers by Lifetime Value', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Lifetime Value ($)')
            ax3.grid(True, alpha=0.3, axis='x')
        
        # Interaction types
        ax4 = fig.add_subplot(gs[2, 0])
        interaction_df = self.interaction_summary()
        if len(interaction_df) > 0:
            ax4.bar(interaction_df['interaction_type'], interaction_df['count'])
            ax4.set_title('Interactions (Last 30 Days)', fontweight='bold', fontsize=12)
            ax4.set_xlabel('Interaction Type')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Revenue by product
        ax5 = fig.add_subplot(gs[2, 1:])
        product_revenue = self.revenue_by_product()
        if len(product_revenue) > 0:
            ax5.bar(product_revenue['product_name'][:10], product_revenue['total_revenue'][:10])
            ax5.set_title('Top Products by Revenue', fontweight='bold', fontsize=12)
            ax5.set_xlabel('Product')
            ax5.set_ylabel('Revenue ($)')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('CRM Dashboard', fontsize=18, fontweight='bold', y=0.995)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved: {output_file}")
        plt.close()


class CRMSystem:
    """Main CRM System integrating all components"""
    
    def __init__(self, db_name: str = 'crm_database.db'):
        """Initialize CRM system"""
        print("="*80)
        print("CUSTOMER RELATIONSHIP MANAGEMENT (CRM) SYSTEM")
        print("="*80)
        
        self.db = CRMDatabase(db_name)
        self.customers = CustomerManager(self.db)
        self.interactions = InteractionManager(self.db)
        self.opportunities = OpportunityManager(self.db)
        self.products = ProductManager(self.db)
        self.transactions = TransactionManager(self.db)
        self.analytics = CRMAnalytics(self.db)
        
        print("\n✓ CRM System initialized successfully")
    
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        print("\n" + "="*80)
        print("GENERATING SAMPLE DATA")
        print("="*80)
        
        # Sample customers
        customers_data = [
            ('John', 'Doe', 'john.doe@example.com', '555-0101', 'Acme Corp', 'Technology', 'Client'),
            ('Jane', 'Smith', 'jane.smith@example.com', '555-0102', 'Tech Solutions', 'Technology', 'Client'),
            ('Bob', 'Johnson', 'bob.johnson@example.com', '555-0103', 'Global Industries', 'Manufacturing', 'Prospect'),
            ('Alice', 'Williams', 'alice.w@example.com', '555-0104', 'StartUp Inc', 'Technology', 'Client'),
            ('Charlie', 'Brown', 'charlie.b@example.com', '555-0105', 'Enterprise Ltd', 'Finance', 'Prospect'),
        ]
        
        customer_ids = []
        for first, last, email, phone, company, industry, ctype in customers_data:
            cid = self.customers.add_customer(first, last, email, phone, company, industry, ctype)
            if cid > 0:
                customer_ids.append(cid)
        
        # Sample products
        products_data = [
            ('CRM Software License', 'Software', 999.99),
            ('Consulting Services', 'Services', 150.00),
            ('Training Workshop', 'Training', 500.00),
            ('Support Package', 'Support', 299.99),
        ]
        
        product_ids = []
        for name, category, price in products_data:
            pid = self.products.add_product(name, category, price)
            if pid > 0:
                product_ids.append(pid)
        
        # Sample transactions
        if customer_ids and product_ids:
            for i in range(min(10, len(customer_ids) * 2)):
                cust_id = np.random.choice(customer_ids)
                prod_id = np.random.choice(product_ids)
                quantity = np.random.randint(1, 5)
                
                # Get product price
                self.db.cursor.execute('SELECT price FROM products WHERE product_id = ?', (prod_id,))
                result = self.db.cursor.fetchone()
                if result:
                    price = result[0]
                    self.transactions.record_transaction(cust_id, prod_id, quantity, price)
        
        # Sample interactions
        interaction_types = ['Email', 'Phone Call', 'Meeting', 'Demo', 'Support']
        for cust_id in customer_ids[:3]:
            for _ in range(np.random.randint(2, 5)):
                itype = np.random.choice(interaction_types)
                self.interactions.log_interaction(
                    cust_id, itype,
                    f"Discussion about {itype.lower()}",
                    description=f"Had a productive {itype.lower()} with customer",
                    outcome='Positive',
                    duration_minutes=np.random.randint(15, 60)
                )
        
        # Sample opportunities
        for cust_id in customer_ids[:3]:
            value = np.random.randint(5000, 50000)
            stage = np.random.choice(['Prospecting', 'Qualification', 'Proposal'])
            expected_date = (datetime.now() + timedelta(days=np.random.randint(30, 90))).strftime('%Y-%m-%d')
            self.opportunities.create_opportunity(
                cust_id, f"Opportunity for Customer {cust_id}",
                value, stage, expected_close_date=expected_date
            )
        
        print("\n✓ Sample data generated successfully")
    
    def print_summary(self):
        """Print CRM summary report"""
        print("\n" + "="*80)
        print("CRM SUMMARY REPORT")
        print("="*80)
        
        # Customer summary
        cust_summary = self.analytics.customer_summary()
        print("\n📊 CUSTOMER METRICS:")
        print(f"  Total Customers: {cust_summary['total_customers']}")
        print(f"  Active Customers: {cust_summary['active_customers']}")
        print(f"  Clients: {cust_summary['clients']}")
        print(f"  Prospects: {cust_summary['prospects']}")
        print(f"  Total Lifetime Value: ${cust_summary['total_lifetime_value']:,.2f}")
        print(f"  Average Lifetime Value: ${cust_summary['avg_lifetime_value']:,.2f}")
        
        # Sales summary
        sales_summary = self.analytics.sales_summary()
        print("\n💰 SALES METRICS:")
        print(f"  Total Transactions: {sales_summary['total_transactions'] or 0}")
        print(f"  Total Revenue: ${sales_summary['total_revenue'] or 0:,.2f}")
        print(f"  Average Transaction: ${sales_summary['avg_transaction_value'] or 0:,.2f}")
        print(f"  Largest Transaction: ${sales_summary['largest_transaction'] or 0:,.2f}")
        
        # Pipeline forecast
        forecast = self.opportunities.get_forecast()
        print("\n📈 SALES PIPELINE:")
        print(f"  Total Pipeline Value: ${forecast['total_pipeline']:,.2f}")
        print(f"  Weighted Forecast: ${forecast['weighted_forecast']:,.2f}")
        print(f"  Best Case: ${forecast['best_case']:,.2f}")
        print(f"  Open Opportunities: {forecast['opportunities_count']}")
        
        # Top customers
        print("\n🏆 TOP 5 CUSTOMERS:")
        top_customers = self.analytics.top_customers(limit=5)
        for idx, row in top_customers.iterrows():
            print(f"  {row['first_name']} {row['last_name']} ({row['company']}) - ${row['lifetime_value']:,.2f}")
    
    def export_reports(self):
        """Export various reports to CSV"""
        print("\n" + "="*80)
        print("EXPORTING REPORTS")
        print("="*80)
        
        # All customers
        customers_df = pd.read_sql_query('SELECT * FROM customers', self.db.conn)
        customers_df.to_csv('crm_customers_report.csv', index=False)
        print(f"✓ Customers report exported: crm_customers_report.csv ({len(customers_df)} records)")
        
        # All interactions
        interactions_df = pd.read_sql_query('SELECT * FROM interactions', self.db.conn)
        interactions_df.to_csv('crm_interactions_report.csv', index=False)
        print(f"✓ Interactions report exported: crm_interactions_report.csv ({len(interactions_df)} records)")
        
        # Sales pipeline
        pipeline_df = self.opportunities.get_pipeline()
        pipeline_df.to_csv('crm_pipeline_report.csv', index=False)
        print(f"✓ Pipeline report exported: crm_pipeline_report.csv ({len(pipeline_df)} records)")
        
        # Revenue by product
        product_revenue = self.analytics.revenue_by_product()
        product_revenue.to_csv('crm_product_revenue_report.csv', index=False)
        print(f"✓ Product revenue report exported: crm_product_revenue_report.csv ({len(product_revenue)} records)")
    
    def close(self):
        """Close CRM system"""
        self.db.close()


def main():
    """Main execution function"""
    
    # Initialize CRM system
    crm = CRMSystem('crm_database.db')
    
    # Generate sample data
    crm.generate_sample_data()
    
    # Print summary
    crm.print_summary()
    
    # Export reports
    crm.export_reports()
    
    # Create dashboard
    print("\n" + "="*80)
    print("CREATING ANALYTICS DASHBOARD")
    print("="*80)
    crm.analytics.create_dashboard('crm_dashboard.png')
    
    # Close system
    print("\n" + "="*80)
    print("CRM SYSTEM SESSION COMPLETE")
    print("="*80)
    crm.close()


if __name__ == "__main__":
    main()
