
my_list = []
n = int(input("Enter the number of elements to add to the list: "))
for i in range(n):
    element = int(input(f"Enter element {i+1}: "))
    my_list.append(element)
print("Initial list:", my_list)

# Adding an element to the list
element_to_add = int(input("Enter an element to append to the list: "))
my_list.append(element_to_add)
print("List after appending:", my_list)

# Removing an element from the list
element_to_remove = int(input("Enter an element to remove from the list: "))
if element_to_remove in my_list:
    my_list.remove(element_to_remove)
    print("List after removing:", my_list)
else:
    print(f"Element {element_to_remove} not found in the list.")

# Modifying an element in the list
index_to_modify = int(input("Enter the index of the element to modify in the list: "))
if 0 <= index_to_modify < len(my_list):
    new_value = int(input("Enter the new value: "))
    my_list[index_to_modify] = new_value
    print("List after modifying:", my_list)
else:
    print(f"Index {index_to_modify} is out of range.")

print("\n")

# Creating a dictionary
my_dict = {}
n = int(input("Enter the number of key-value pairs to add to the dictionary: "))
for i in range(n):
    key = input(f"Enter key {i+1}: ")
    value = int(input(f"Enter value for key '{key}': "))
    my_dict[key] = value
print("Initial dictionary:", my_dict)

# Adding an element to the dictionary
new_key = input("Enter a new key to add to the dictionary: ")
new_value = int(input(f"Enter value for key '{new_key}': "))
my_dict[new_key] = new_value
print("Dictionary after adding:", my_dict)

# Removing an element from the dictionary
key_to_remove = input("Enter the key to remove from the dictionary: ")
if key_to_remove in my_dict:
    del my_dict[key_to_remove]
    print("Dictionary after removing:", my_dict)
else:
    print(f"Key '{key_to_remove}' not found in the dictionary.")

# Modifying an element in the dictionary
key_to_modify = input("Enter the key to modify in the dictionary: ")
if key_to_modify in my_dict:
    new_value = int(input(f"Enter the new value for key '{key_to_modify}': "))
    my_dict[key_to_modify] = new_value
    print("Dictionary after modifying:", my_dict)
else:
    print(f"Key '{key_to_modify}' not found in the dictionary.")

print("\n")

# Creating a set
my_set = set()
n = int(input("Enter the number of elements to add to the set: "))
for i in range(n):
    element = int(input(f"Enter element {i+1}: "))
    my_set.add(element)
print("Initial set:", my_set)

# Adding an element to the set
element_to_add = int(input("Enter an element to add to the set: "))
my_set.add(element_to_add)
print("Set after adding:", my_set)

# Removing an element from the set
element_to_remove = int(input("Enter an element to remove from the set: "))
if element_to_remove in my_set:
    my_set.remove(element_to_remove)
    print("Set after removing:", my_set)
else:
    print(f"Element {element_to_remove} not found in the set.")

# Modifying an element in the set
element_to_modify = int(input("Enter an element to modify in the set: "))
if element_to_modify in my_set:
    new_value = int(input("Enter the new value: "))
    my_set.remove(element_to_modify)
    my_set.add(new_value)
    print("Set after modifying:", my_set)
else:
    print(f"Element {element_to_modify} not found in the set.")
