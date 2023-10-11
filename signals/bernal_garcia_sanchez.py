import tkinter as tk
import subprocess


# Function to create a new window with Ejercicio 1 and Ejercicio 2 buttons
def create_second_window():
    second_window = tk.Toplevel()
    second_window.title("Second Window")
    second_window.configure(bg="black")

    # Create Ejercicio 1 button in the new window
    ejercicio1_button = tk.Button(
        second_window,
        text="Ejercicio 1",
        command=create_third_window,  # Define the action for this button
    )
    ejercicio1_button.pack(pady=5)

    # Create Ejercicio 2 button in the new window
    ejercicio2_button = tk.Button(
        second_window,
        text="Ejercicio 2",
        command=create_fourth_window,  # Replace with your action for Ejercicio 2
    )
    ejercicio2_button.pack(pady=5)


# Function to create a new window when Ejercicio 1 is pressed
def create_third_window():
    third_window = tk.Toplevel()
    third_window.title("Third Window")
    third_window.configure(bg="black")

    # Add label for instruction
    instruction_label = tk.Label(
        third_window,
        text="Escoger función a girar y mover",
        font=("Helvetica", 16),
        fg="white",
        bg="black",
    )
    instruction_label.pack(pady=10)

    def x_n_action():
        try:
            # Execute the "h[n].py" file without specifying a path
            subprocess.run(["python", "xn.py"])
        except FileNotFoundError:
            print("Error: The 'xn.py' file was not found.")

    # Create two buttons "x[n]" and "h[n]"
    x_n_button = tk.Button(
        third_window,
        text="x[n]",
        command=x_n_action,  # Replace with your action for x[n]
    )
    x_n_button.pack(pady=5)

    # Function to be called when "h[n]" button is clicked
    def h_n_action():
        try:
            # Execute the "h[n].py" file without specifying a path
            subprocess.run(["python", "hn.py"])
        except FileNotFoundError:
            print("Error: The 'hn.py' file was not found.")

    h_n_button = tk.Button(
        third_window,
        text="h[n]",
        command=h_n_action,
    )
    h_n_button.pack(pady=5)


# Function to create a new window when Ejercicio 2 is pressed
def create_fourth_window():
    fourth_window = tk.Toplevel()
    fourth_window.title("Fourth Window")
    fourth_window.configure(bg="black")

    # Add label for instruction
    instruction_label = tk.Label(
        fourth_window,
        text="Escoger función a girar y mover",
        font=("Helvetica", 16),
        fg="white",
        bg="black",
    )
    instruction_label.pack(pady=10)

    def x_n_action():
        try:
            # Execute the "h[n].py" file without specifying a path
            subprocess.run(["python", "xn2.py"])
        except FileNotFoundError:
            print("Error: The 'xn2.py' file was not found.")

    # Create two buttons "x[n]" and "h[n]"
    x_n_button = tk.Button(
        fourth_window,
        text="x[n]",
        command=x_n_action,  # Replace with your action for x[n]
    )
    x_n_button.pack(pady=5)

    # Function to be called when "h[n]" button is clicked
    def h_n_action():
        try:
            # Execute the "h[n].py" file without specifying a path
            subprocess.run(["python", "hn2.py"])
        except FileNotFoundError:
            print("Error: The 'hn2.py' file was not found.")

    h_n_button = tk.Button(
        fourth_window,
        text="h[n]",
        command=h_n_action,
    )
    h_n_button.pack(pady=5)


# Function to be called when Discrete Time button is clicked
def discrete_time_action():
    # Create a new window with Ejercicio 1 and Ejercicio 2 buttons
    create_second_window()


# Function to be called when Continuous Time button is clicked
def continuous_time_action():
    # Replace with your action for Continuous Time
    pass


# Create a main window
root = tk.Tk()
root.title("Bernal Garcia Sanchez")
root.configure(bg="black")

# Create a label with your name
label = tk.Label(
    root,
    text="Laboratorio de Convolución",
    font=("Helvetica", 24),
    fg="white",
    bg="black",
)
label.grid(row=0, pady=10)

# Center the label
label.grid_configure(sticky="nsew")

# Create Discrete Time button
discrete_time_button = tk.Button(
    root,
    text="Tiempo Discreto",
    command=discrete_time_action,  # Define the action for this button
)
discrete_time_button.grid(row=1, pady=5)

# Create Continuous Time button
continuous_time_button = tk.Button(
    root,
    text="Tiempo Continuo",
    command=continuous_time_action,  # Define the action for this button
)
continuous_time_button.grid(row=2, pady=5)

# Create a label for names
names_label = tk.Label(
    root,
    text="Cristian Bernal, José García y Juan Sánchez",
    font=("Helvetica", 18),
    fg="white",
    bg="black",
)
names_label.grid(row=3, pady=10)

# Configure grid rows and columns to expand
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Start the GUI application
root.geometry("1600x900")
root.mainloop()
