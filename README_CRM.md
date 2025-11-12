# 📊 Customer Relationship Management (CRM) System

**Professional CRM Solution with SQLite Database, Analytics & Reporting**

---

## 🎯 System Overview

A complete Customer Relationship Management system built in Python featuring:

- **Customer Management** - Track contacts, companies, and customer types
- **Interaction Logging** - Record all customer touchpoints (calls, emails, meetings)
- **Sales Pipeline** - Manage opportunities from prospecting to close
- **Product Catalog** - Maintain products/services with pricing
- **Transaction Tracking** - Record sales and calculate lifetime value
- **Analytics Dashboard** - Visual insights and KPI tracking
- **Report Generation** - Export data to CSV for further analysis

---

## 🗄️ Database Schema

### Tables Created

1. **`customers`** - Core customer information
   - customer_id, first_name, last_name, email, phone
   - company, industry, customer_type, status
   - lifetime_value, created_date, last_contact_date

2. **`interactions`** - Customer touchpoint log
   - interaction_id, customer_id, interaction_type
   - interaction_date, subject, description, outcome
   - next_action, follow_up_date, duration_minutes

3. **`opportunities`** - Sales pipeline
   - opportunity_id, customer_id, opportunity_name
   - value, stage, probability, expected_close_date
   - status, created_date, closed_date

4. **`products`** - Product/service catalog
   - product_id, product_name, category, price
   - description, active

5. **`transactions`** - Sales records
   - transaction_id, customer_id, product_id
   - transaction_date, quantity, unit_price, total_amount
   - payment_method, status

6. **`tasks`** - Task management
   - task_id, customer_id, task_title, description
   - assigned_to, priority, due_date, status

---

## 💡 Key Features

### Customer Management
- ✅ Add, update, search, and delete customers
- ✅ Track customer type (Prospect, Client, etc.)
- ✅ Monitor customer status (Active/Inactive)
- ✅ Calculate lifetime value automatically
- ✅ Search with multiple filters

### Interaction Tracking
- ✅ Log all customer interactions (Email, Phone, Meeting, Demo, Support)
- ✅ Record outcomes and next actions
- ✅ Set follow-up dates
- ✅ Track interaction duration
- ✅ View customer interaction history

### Sales Pipeline Management
- ✅ Create and track opportunities
- ✅ Move through sales stages (Prospecting → Qualification → Proposal → Negotiation → Closed)
- ✅ Automatic probability calculation by stage
- ✅ Expected close date tracking
- ✅ Win/loss status monitoring

### Sales Forecasting
- ✅ Total pipeline value
- ✅ Weighted forecast (value × probability)
- ✅ Best case scenario (high probability deals)
- ✅ Opportunity count tracking

### Product Management
- ✅ Maintain product catalog with pricing
- ✅ Categorize products/services
- ✅ Active/inactive product status
- ✅ Product descriptions

### Transaction Recording
- ✅ Record customer purchases
- ✅ Automatic lifetime value calculation
- ✅ Multiple payment method support
- ✅ Transaction history per customer

### Analytics & Reporting
- ✅ Customer summary statistics
- ✅ Sales metrics and KPIs
- ✅ Top customers by lifetime value
- ✅ Interaction analysis
- ✅ Revenue by product breakdown
- ✅ Visual dashboard creation

---

## 📊 Analytics Capabilities

### Customer Metrics
- Total customers count
- Active vs inactive customers
- Client vs prospect breakdown
- Total lifetime value
- Average lifetime value per customer

### Sales Metrics
- Total transactions
- Total revenue
- Average transaction value
- Largest transaction
- Revenue trends

### Pipeline Metrics
- Total pipeline value
- Weighted forecast
- Best case revenue
- Open opportunities count
- Win rate tracking

### Dashboard Visualizations
1. Customer type distribution (pie chart)
2. Revenue trend (line chart, last 30 days)
3. Top 10 customers by lifetime value (bar chart)
4. Interaction types breakdown (bar chart)
5. Top products by revenue (bar chart)

---

## 🚀 Usage

### Basic Workflow

```python
from crm_system import CRMSystem

# Initialize CRM
crm = CRMSystem('my_crm.db')

# Add a customer
customer_id = crm.customers.add_customer(
    first_name='John',
    last_name='Doe',
    email='john.doe@company.com',
    phone='555-1234',
    company='Acme Corp',
    industry='Technology',
    customer_type='Prospect'
)

# Log an interaction
crm.interactions.log_interaction(
    customer_id=customer_id,
    interaction_type='Phone Call',
    subject='Initial outreach',
    description='Discussed product demo',
    outcome='Positive',
    follow_up_date='2025-11-20',
    duration_minutes=30
)

# Create sales opportunity
crm.opportunities.create_opportunity(
    customer_id=customer_id,
    opportunity_name='CRM Software Deal',
    value=15000.00,
    stage='Qualification',
    expected_close_date='2025-12-31'
)

# Add product
product_id = crm.products.add_product(
    product_name='CRM License',
    category='Software',
    price=999.99,
    description='Annual CRM software license'
)

# Record transaction
crm.transactions.record_transaction(
    customer_id=customer_id,
    product_id=product_id,
    quantity=5,
    unit_price=999.99,
    payment_method='Credit Card'
)

# Get analytics
summary = crm.analytics.customer_summary()
print(summary)

# Create dashboard
crm.analytics.create_dashboard('my_dashboard.png')

# Export reports
crm.export_reports()

# Close
crm.close()
```

### Search & Filter

```python
# Search customers
results = crm.customers.search_customers(
    search_term='tech',
    customer_type='Client',
    status='Active'
)

# Get top customers
top_customers = crm.analytics.top_customers(limit=10)

# Get pipeline
pipeline = crm.opportunities.get_pipeline()

# Get forecast
forecast = crm.opportunities.get_forecast()
```

---

## 📁 Files Generated

### Database
- `crm_database.db` - SQLite database with all CRM data

### Reports (CSV)
- `crm_customers_report.csv` - All customer records
- `crm_interactions_report.csv` - All interactions
- `crm_pipeline_report.csv` - Open opportunities
- `crm_product_revenue_report.csv` - Revenue by product

### Visualizations
- `crm_dashboard.png` - Comprehensive analytics dashboard (300 DPI)

---

## 🎨 Dashboard Components

The auto-generated dashboard (`crm_dashboard.png`) includes:

1. **Customer Type Distribution** - Pie chart showing Clients vs Prospects
2. **Revenue Trend** - 30-day revenue line chart
3. **Top 10 Customers** - Horizontal bar chart by lifetime value
4. **Interaction Types** - Bar chart of recent interactions (30 days)
5. **Top Products** - Revenue by product bar chart

---

## 📈 Sales Stages & Probabilities

| Stage | Default Probability | Description |
|-------|---------------------|-------------|
| **Prospecting** | 10% | Initial contact |
| **Qualification** | 25% | Qualified lead |
| **Proposal** | 50% | Proposal sent |
| **Negotiation** | 75% | In negotiations |
| **Closed Won** | 100% | Deal won |
| **Closed Lost** | 0% | Deal lost |

---

## 🔧 Technical Specifications

### Technology Stack
- **Language:** Python 3.13+
- **Database:** SQLite3
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Type Hints:** Full typing support

### Database Design
- **Relational model** with foreign key constraints
- **Indexed primary keys** for performance
- **Unique constraints** on email, product names
- **Automatic timestamps** for audit trails
- **Soft deletes** (status changes, not deletions)

### Performance
- Optimized queries with proper indexing
- Batch operations supported
- Transaction safety with commits
- Scalable to 10,000+ customers

---

## 📊 Sample Output

```
================================================================================
CRM SUMMARY REPORT
================================================================================

📊 CUSTOMER METRICS:
  Total Customers: 5
  Active Customers: 5
  Clients: 3
  Prospects: 2
  Total Lifetime Value: $12,450.00
  Average Lifetime Value: $2,490.00

💰 SALES METRICS:
  Total Transactions: 15
  Total Revenue: $12,450.00
  Average Transaction: $830.00
  Largest Transaction: $2,999.97

📈 SALES PIPELINE:
  Total Pipeline Value: $106,950.00
  Weighted Forecast: $32,085.00
  Best Case: $45,000.00
  Open Opportunities: 3

🏆 TOP 5 CUSTOMERS:
  John Doe (Acme Corp) - $4,999.95
  Jane Smith (Tech Solutions) - $3,499.96
  Alice Williams (StartUp Inc) - $2,499.98
  Bob Johnson (Global Industries) - $1,000.00
  Charlie Brown (Enterprise Ltd) - $450.11
```

---

## 🎯 Use Cases

### Sales Team
- Track all customer interactions
- Manage sales pipeline
- Forecast revenue
- Identify top customers
- Follow up on opportunities

### Marketing Team
- Segment customers by type/industry
- Analyze interaction patterns
- Identify cross-sell opportunities
- Track customer lifetime value
- Generate target lists

### Management
- Monitor sales performance
- Track KPIs and metrics
- Forecast revenue
- Analyze customer trends
- Export reports for presentations

### Customer Support
- View customer history
- Log support interactions
- Track follow-ups
- Update customer status
- Monitor satisfaction

---

## 🔐 Data Security

- **No hardcoded credentials**
- **Local SQLite database** (can be upgraded to PostgreSQL/MySQL)
- **Parameterized queries** (SQL injection protection)
- **Data validation** on input
- **Audit trail** with timestamps

---

## 🚀 Future Enhancements

Potential additions for enterprise deployment:

- [ ] User authentication & authorization
- [ ] Role-based access control (RBAC)
- [ ] Email integration (send/receive)
- [ ] Calendar synchronization
- [ ] Document attachment storage
- [ ] Custom fields & workflows
- [ ] API for external integrations
- [ ] Web interface (Flask/Django)
- [ ] Mobile app compatibility
- [ ] Multi-user concurrent access
- [ ] Advanced reporting (pivot tables)
- [ ] Machine learning predictions
- [ ] Customer churn analysis
- [ ] Marketing automation
- [ ] Quote generation

---

## 📝 License

Open source - Free for personal and commercial use

---

## 👨‍💻 Created By

Part of the comprehensive data analysis and business intelligence toolkit

**Repository:** https://github.com/Olebogeng3/Unsupervised-Learning-Project  
**Date:** November 12, 2025

---

## 🎓 Learning Outcomes

This CRM system demonstrates:

✅ **Database Design** - Relational schema with proper normalization  
✅ **Object-Oriented Programming** - Clean class-based architecture  
✅ **Data Analytics** - KPI calculation and business metrics  
✅ **Data Visualization** - Dashboard creation with matplotlib  
✅ **SQL Operations** - CRUD operations and complex queries  
✅ **Python Best Practices** - Type hints, docstrings, error handling  
✅ **Business Intelligence** - Real-world CRM functionality

---

## 📞 Support

For questions or issues:
- Review the code documentation (extensive docstrings)
- Check the database schema
- Examine the sample data generation
- Test with the demo workflow

---

**🎉 Ready to manage your customer relationships professionally!**
