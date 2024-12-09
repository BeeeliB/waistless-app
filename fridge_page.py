import streamlit as st # Streamlit for building the user interface
import pandas as pd # Library to handle data
from datetime import datetime # To handle timestamps for purchases and consumption

# Initialization of the session status for saving values between interactions, just for testing
if "roommates" not in st.session_state:
    st.session_state["roommates"] = ["Livio", "Flurin", "Anderin"] # Default roommates for testing
if "inventory" not in st.session_state: 
    st.session_state["inventory"] = {} # Dictionary to store inventory data
if "expenses" not in st.session_state:
    # Dictionary to track total expenses for each roommate
    st.session_state["expenses"] = {mate: 0.0 for mate in st.session_state["roommates"]}
if "purchases" not in st.session_state:
    # Dictionary to log purchases made by each roommate
    st.session_state["purchases"] = {mate: [] for mate in st.session_state["roommates"]}
if "consumed" not in st.session_state:
    # Dictionary to log consumed items for each roommate
    st.session_state["consumed"] = {mate: [] for mate in st.session_state["roommates"]}

# Ensure that entries in expenses, purchases and consumption are initialized when adding or removing roommates
def ensure_roommate_entries():
    for mate in st.session_state["roommates"]:
        if mate not in st.session_state["expenses"]: # Add missing expense entry
            st.session_state["expenses"][mate] = 0.0
        if mate not in st.session_state["purchases"]: # Add missing purchase log
            st.session_state["purchases"][mate] = []
        if mate not in st.session_state["consumed"]: # Add missing consumption log
            st.session_state["consumed"][mate] = []

# Function to remove product from inventory
def delete_product_from_inventory(food_item, quantity, unit, selected_roommate):
    ensure_roommate_entries() # Ensure all roommate-related data is initialized
    delete_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Records the current time
    
    if food_item and quantity > 0 and selected_roommate:
        # Check if the item already exists in inventory
        if food_item in st.session_state["inventory"]: 
            current_quantity = st.session_state["inventory"][food_item]["Quantity"]
            current_price = st.session_state["inventory"][food_item]["Price"]
            # Update the quantity and total price of the existing item
            if quantity <= current_quantity:
                # Calculate the price
                price_per_unit = current_price / current_quantity if current_quantity > 0 else 0
                amount_to_deduct = price_per_unit * quantity
                # Update inventory
                st.session_state["inventory"][food_item]["Quantity"] -= quantity
                st.session_state["inventory"][food_item]["Price"] -= amount_to_deduct
                st.session_state["expenses"][selected_roommate] -= amount_to_deduct
                st.success(f"'{quantity}' of '{food_item}' has been removed.")
                # Report the ingredients in consumed
                st.session_state["consumed"][selected_roommate].append({
                    "Product": food_item,
                    "Quantity": quantity,
                    "Price": amount_to_deduct,
                    "Unit": unit,
                    "Date": delete_time
                })
                # Remove item if quantity reaches zero
                if st.session_state["inventory"][food_item]["Quantity"] <= 0:
                    del st.session_state["inventory"][food_item]
            else:
                st.warning("The quantity to remove exceeds the available quantity.") # Warning message
        else:
            st.warning("This item is not in the inventory.") # Warning message
    else:
        st.warning("Please fill in all fields.") # Warning message

# Function to add product to inventory
def add_product_to_inventory(food_item, quantity, unit, price, selected_roommate):
    ensure_roommate_entries()
    purchase_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if food_item in st.session_state["inventory"]:  # checks if the food is already in the inventory
        st.session_state["inventory"][food_item]["Quantity"] += quantity
        st.session_state["inventory"][food_item]["Price"] += price
    else:
        st.session_state["inventory"][food_item] = {"Quantity": quantity, "Unit": unit, "Price": price}
    
    st.session_state["expenses"][selected_roommate] += price
    st.session_state["purchases"][selected_roommate].append({
        "Product": food_item,
        "Quantity": quantity,
        "Price": price,
        "Unit": unit,
        "Date": purchase_time
    })
    st.success(f"'{food_item}' has been added to the inventory, and {selected_roommate}'s expenses were updated.")

# Main page function
def fridge_page():
    ensure_roommate_entries() # Ensure roommate-related data is ready
    st.title("Inventory")  # Display the page title

    # Roommate selection
    if st.session_state["roommates"]:
        selected_roommate = st.selectbox("Select the roommate:", st.session_state["roommates"])
    else:
        st.warning("No roommates available.")
        return # Exit the function if no roommates are defined

    # Selection: Add or remove item from inventory
    action = st.selectbox("Would you like to add or remove an item?", ["Add", "Remove"])

    # If "Add" is selected, display input fields for adding an item
    if action == "Add":
        # Input fields for food item, quantity, unit, and price
        food_item = st.selectbox("Select a food item to add:", [
            'chicken', 'curry powder', 'coconut milk', 'onion', 'garlic', 'ginger',
            'beef', 'potatoes', 'carrots', 'onions', 'beef broth',
            'broccoli', 'bell peppers', 'soy sauce', 'tofu',
            'lentils', 'celery', 'vegetable broth',
            'fish', 'tortillas', 'cabbage', 'lime', 'avocado', 'salsa',
            'eggs', 'cream', 'bacon', 'cheese', 'pie crust',
            'romaine lettuce', 'croutons', 'parmesan', 'caesar dressing',
            'flour', 'sugar', 'cocoa powder', 'butter', 'baking powder',
            'apples', 'cinnamon', 'lemon juice',
            'bread', 'parsley'
        ])
        quantity = st.number_input("Quantity:", min_value=0.0)
        unit = st.selectbox("Unit:", ["Pieces", "Liters", "Grams"])
        price = st.number_input("Price (in CHF):", min_value=0.0)
        
        if st.button("Add item"): # Button to confirm adding the item
            if food_item and quantity > 0 and price >= 0 and selected_roommate:
                add_product_to_inventory(food_item, quantity, unit, price, selected_roommate)
            else:
                st.warning("Please fill in all fields.")
    
    elif action == "Remove": # If "Remove" is selected, display input fields for removing an item
        if st.session_state["inventory"]:
            # Selection of the food and quantity to be removed
            food_item = st.selectbox("Select a food item to remove:", list(st.session_state["inventory"].keys()))
            quantity = st.number_input("Quantity to remove:", min_value=1.0, step=1.0)
            unit = st.session_state["inventory"][food_item]["Unit"]
            if st.button("Remove item"): # Button to confirm removing the item
                delete_product_from_inventory(food_item, quantity, unit, selected_roommate)
        else:
            st.warning("The inventory is empty.")

    # Display current inventory
    if st.session_state["inventory"]:
        st.write("Current Inventory:")
        inventory_df = pd.DataFrame.from_dict(st.session_state["inventory"], orient='index') # Creates a DataFrame and sets food items as row labels
        inventory_df = inventory_df.reset_index().rename(columns={'index': 'Food Item'}) # Move food item to the second column and rename the column title
        st.table(inventory_df)
    else:
        st.write("The inventory is empty.")

    # Display total expenses per roommate
    st.write("Total expenses per roommate:")
    expenses_df = pd.DataFrame(list(st.session_state["expenses"].items()), columns=["Roommate", "Total Expenses (CHF)"]) #Generates a list of tuples and assigns column titles
    st.table(expenses_df)

    # Display purchases and consumed items per roommate
    st.write("Purchases and Consumptions per roommate:")
    for mate in st.session_state["roommates"]:
        st.write(f"{mate}'s Purchases:")
        purchases_df = pd.DataFrame(st.session_state["purchases"][mate])
        st.table(purchases_df)
        
        st.write(f"{mate}'s Consumptions:")
        consumed_df = pd.DataFrame(st.session_state["consumed"][mate])
        st.table(consumed_df)

# Call the function to display the fridge page
fridge_page()