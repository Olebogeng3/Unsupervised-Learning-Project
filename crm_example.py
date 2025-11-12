"""
CRM System - Quick Start Example
Demonstrates key CRM features with practical examples
"""

from crm_system import CRMSystem
from datetime import datetime, timedelta

def main():
    print("="*80)
    print("CRM SYSTEM - QUICK START GUIDE")
    print("="*80)
    
    # Initialize CRM (creates new database)
    crm = CRMSystem('example_crm.db')
    
    print("\n" + "="*80)
    print("EXAMPLE 1: ADDING CUSTOMERS")
    print("="*80)
    
    # Add customers
    customer1 = crm.customers.add_customer(
        first_name='Sarah',
        last_name='Connor',
        email='sarah.connor@cyberdyne.com',
        phone='555-2001',
        company='Cyberdyne Systems',
        industry='Technology',
        customer_type='Prospect',
        notes='Interested in enterprise solution'
    )
    
    customer2 = crm.customers.add_customer(
        first_name='Tony',
        last_name='Stark',
        email='tony@starkindustries.com',
        phone='555-3000',
        company='Stark Industries',
        industry='Manufacturing',
        customer_type='Client',
        notes='Long-term client, prefers premium products'
    )
    
    customer3 = crm.customers.add_customer(
        first_name='Bruce',
        last_name='Wayne',
        email='bruce.wayne@wayneenterprises.com',
        phone='555-4000',
        company='Wayne Enterprises',
        industry='Conglomerate',
        customer_type='Client'
    )
    
    print("\n" + "="*80)
    print("EXAMPLE 2: ADDING PRODUCTS")
    print("="*80)
    
    # Add products
    product1 = crm.products.add_product(
        product_name='Enterprise CRM Suite',
        category='Software',
        price=4999.99,
        description='Complete CRM solution for large enterprises'
    )
    
    product2 = crm.products.add_product(
        product_name='Professional Training',
        category='Training',
        price=1500.00,
        description='2-day on-site training program'
    )
    
    product3 = crm.products.add_product(
        product_name='Premium Support (Annual)',
        category='Support',
        price=2400.00,
        description='24/7 premium support with 2-hour response time'
    )
    
    print("\n" + "="*80)
    print("EXAMPLE 3: LOGGING INTERACTIONS")
    print("="*80)
    
    # Log interactions
    crm.interactions.log_interaction(
        customer_id=customer1,
        interaction_type='Phone Call',
        subject='Product inquiry',
        description='Customer called to learn about Enterprise CRM Suite. Very interested in features.',
        outcome='Positive - Scheduled demo',
        next_action='Send product brochure and schedule demo',
        follow_up_date=(datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
        duration_minutes=25
    )
    
    crm.interactions.log_interaction(
        customer_id=customer2,
        interaction_type='Meeting',
        subject='Quarterly business review',
        description='Reviewed usage stats and discussed expansion opportunities',
        outcome='Excellent - Interest in additional licenses',
        next_action='Send proposal for 10 additional licenses',
        follow_up_date=(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
        duration_minutes=60
    )
    
    crm.interactions.log_interaction(
        customer_id=customer2,
        interaction_type='Email',
        subject='Support ticket resolution',
        description='Resolved configuration issue with dashboard widgets',
        outcome='Resolved',
        duration_minutes=15
    )
    
    print("\n" + "="*80)
    print("EXAMPLE 4: CREATING SALES OPPORTUNITIES")
    print("="*80)
    
    # Create opportunities
    opp1 = crm.opportunities.create_opportunity(
        customer_id=customer1,
        opportunity_name='Cyberdyne - Enterprise CRM Implementation',
        value=25000.00,
        stage='Qualification',
        probability=25,
        expected_close_date='2025-12-15'
    )
    
    opp2 = crm.opportunities.create_opportunity(
        customer_id=customer2,
        opportunity_name='Stark Industries - License Expansion',
        value=15000.00,
        stage='Proposal',
        probability=50,
        expected_close_date='2025-11-30'
    )
    
    opp3 = crm.opportunities.create_opportunity(
        customer_id=customer3,
        opportunity_name='Wayne Enterprises - Support Renewal',
        value=2400.00,
        stage='Negotiation',
        probability=75,
        expected_close_date='2025-11-25'
    )
    
    print("\n" + "="*80)
    print("EXAMPLE 5: RECORDING TRANSACTIONS")
    print("="*80)
    
    # Record transactions for existing client
    crm.transactions.record_transaction(
        customer_id=customer2,
        product_id=product1,
        quantity=1,
        unit_price=4999.99,
        payment_method='Wire Transfer'
    )
    
    crm.transactions.record_transaction(
        customer_id=customer2,
        product_id=product2,
        quantity=2,
        unit_price=1500.00,
        payment_method='Credit Card'
    )
    
    crm.transactions.record_transaction(
        customer_id=customer3,
        product_id=product3,
        quantity=1,
        unit_price=2400.00,
        payment_method='Check'
    )
    
    print("\n" + "="*80)
    print("EXAMPLE 6: SEARCHING & FILTERING")
    print("="*80)
    
    # Search for technology companies
    print("\n🔍 Searching for 'technology' customers...")
    tech_customers = crm.customers.search_customers(search_term='technology')
    print(f"Found {len(tech_customers)} technology-related customers")
    
    # Get all clients
    print("\n🔍 Filtering for active clients...")
    clients = crm.customers.search_customers(customer_type='Client', status='Active')
    print(f"Found {len(clients)} active clients")
    for idx, row in clients.iterrows():
        print(f"  - {row['first_name']} {row['last_name']} ({row['company']})")
    
    print("\n" + "="*80)
    print("EXAMPLE 7: VIEWING CUSTOMER DETAILS")
    print("="*80)
    
    # Get customer details
    customer_details = crm.customers.get_customer(customer2)
    print(f"\n📋 Customer Details for ID {customer2}:")
    print(f"  Name: {customer_details['first_name']} {customer_details['last_name']}")
    print(f"  Email: {customer_details['email']}")
    print(f"  Company: {customer_details['company']}")
    print(f"  Industry: {customer_details['industry']}")
    print(f"  Type: {customer_details['customer_type']}")
    print(f"  Status: {customer_details['status']}")
    print(f"  Lifetime Value: ${customer_details['lifetime_value']:,.2f}")
    print(f"  Last Contact: {customer_details['last_contact_date']}")
    
    # Get customer interactions
    print(f"\n💬 Interactions for {customer_details['first_name']} {customer_details['last_name']}:")
    interactions = crm.interactions.get_customer_interactions(customer2)
    for idx, row in interactions.iterrows():
        print(f"  - {row['interaction_date']}: {row['interaction_type']} - {row['subject']}")
    
    # Get customer transactions
    print(f"\n💰 Transactions for {customer_details['first_name']} {customer_details['last_name']}:")
    transactions = crm.transactions.get_customer_transactions(customer2)
    for idx, row in transactions.iterrows():
        print(f"  - {row['transaction_date']}: {row['product_name']} × {row['quantity']} = ${row['total_amount']:,.2f}")
    
    print("\n" + "="*80)
    print("EXAMPLE 8: SALES PIPELINE & FORECAST")
    print("="*80)
    
    # View pipeline
    print("\n📊 Current Sales Pipeline:")
    pipeline = crm.opportunities.get_pipeline()
    for idx, row in pipeline.iterrows():
        print(f"  - {row['opportunity_name']}")
        print(f"    Value: ${row['value']:,.2f} | Stage: {row['stage']} | Probability: {row['probability']}%")
        print(f"    Expected Close: {row['expected_close_date']}")
    
    # Get forecast
    print("\n📈 Sales Forecast:")
    forecast = crm.opportunities.get_forecast()
    print(f"  Total Pipeline Value: ${forecast['total_pipeline']:,.2f}")
    print(f"  Weighted Forecast: ${forecast['weighted_forecast']:,.2f}")
    print(f"  Best Case Scenario: ${forecast['best_case']:,.2f}")
    print(f"  Open Opportunities: {forecast['opportunities_count']}")
    
    print("\n" + "="*80)
    print("EXAMPLE 9: ANALYTICS & METRICS")
    print("="*80)
    
    # Customer summary
    cust_summary = crm.analytics.customer_summary()
    print("\n📊 Customer Metrics:")
    print(f"  Total Customers: {cust_summary['total_customers']}")
    print(f"  Active Customers: {cust_summary['active_customers']}")
    print(f"  Clients: {cust_summary['clients']}")
    print(f"  Prospects: {cust_summary['prospects']}")
    print(f"  Total Lifetime Value: ${cust_summary['total_lifetime_value']:,.2f}")
    print(f"  Average Lifetime Value: ${cust_summary['avg_lifetime_value']:,.2f}")
    
    # Sales summary
    sales_summary = crm.analytics.sales_summary()
    print("\n💰 Sales Metrics:")
    print(f"  Total Transactions: {sales_summary['total_transactions'] or 0}")
    print(f"  Total Revenue: ${sales_summary['total_revenue'] or 0:,.2f}")
    print(f"  Average Transaction: ${sales_summary['avg_transaction_value'] or 0:,.2f}")
    
    # Top customers
    print("\n🏆 Top Customers:")
    top_customers = crm.analytics.top_customers(limit=5)
    for idx, row in top_customers.iterrows():
        print(f"  {idx+1}. {row['first_name']} {row['last_name']} ({row['company']}) - ${row['lifetime_value']:,.2f}")
    
    # Interaction summary
    print("\n💬 Interaction Summary (Last 30 Days):")
    interaction_summary = crm.analytics.interaction_summary()
    for idx, row in interaction_summary.iterrows():
        print(f"  {row['interaction_type']}: {row['count']} interactions (Avg {row['avg_duration']:.1f} min)")
    
    # Revenue by product
    print("\n📦 Revenue by Product:")
    product_revenue = crm.analytics.revenue_by_product()
    for idx, row in product_revenue.iterrows():
        print(f"  {row['product_name']} ({row['category']}): {row['transactions']} sales = ${row['total_revenue']:,.2f}")
    
    print("\n" + "="*80)
    print("EXAMPLE 10: UPDATING OPPORTUNITIES")
    print("="*80)
    
    # Update opportunity stage
    print(f"\n✏️ Moving opportunity {opp3} to 'Closed Won'...")
    crm.opportunities.update_opportunity_stage(opp3, 'Closed Won')
    
    # Check updated forecast
    forecast_after = crm.opportunities.get_forecast()
    print(f"\n📈 Updated Forecast:")
    print(f"  Total Pipeline Value: ${forecast_after['total_pipeline']:,.2f}")
    print(f"  Weighted Forecast: ${forecast_after['weighted_forecast']:,.2f}")
    print(f"  Open Opportunities: {forecast_after['opportunities_count']}")
    
    print("\n" + "="*80)
    print("EXAMPLE 11: UPCOMING FOLLOW-UPS")
    print("="*80)
    
    # Get upcoming follow-ups
    print("\n📅 Follow-ups in Next 7 Days:")
    followups = crm.interactions.get_upcoming_followups(days_ahead=7)
    if len(followups) > 0:
        for idx, row in followups.iterrows():
            print(f"  - {row['follow_up_date']}: {row['first_name']} {row['last_name']}")
            print(f"    Action: {row['next_action']}")
            print(f"    Contact: {row['email']} | {row['phone']}")
    else:
        print("  No follow-ups scheduled in the next 7 days")
    
    print("\n" + "="*80)
    print("EXAMPLE 12: EXPORTING REPORTS")
    print("="*80)
    
    # Export reports
    crm.export_reports()
    
    print("\n" + "="*80)
    print("EXAMPLE 13: CREATING DASHBOARD")
    print("="*80)
    
    # Create visual dashboard
    crm.analytics.create_dashboard('example_crm_dashboard.png')
    
    print("\n" + "="*80)
    print("QUICK START COMPLETE!")
    print("="*80)
    print("\n✅ Database created: example_crm.db")
    print("✅ Reports exported: crm_*_report.csv")
    print("✅ Dashboard created: example_crm_dashboard.png")
    print("\n💡 You can now:")
    print("   - Open the database with SQLite browser")
    print("   - View reports in Excel/Google Sheets")
    print("   - Check the dashboard visualization")
    print("   - Modify the code for your specific needs")
    
    # Close CRM
    crm.close()

if __name__ == "__main__":
    main()
