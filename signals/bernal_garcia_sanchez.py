import tkinter as tk
import subprocess
from tkinter import Entry


# Función para crear una nueva ventana con botones de Ejercicio 1 y Ejercicio 2
def crear_segunda_ventana():
    segunda_ventana = tk.Toplevel()
    segunda_ventana.title("Segunda Ventana")
    segunda_ventana.configure(bg="black")

    # Crear el botón Ejercicio 1 en la nueva ventana
    ejercicio1_button = tk.Button(
        segunda_ventana,
        text="Ejercicio 1",
        command=crear_tercera_ventana,  # Definir la acción para este botón
    )
    ejercicio1_button.pack(pady=5)

    # Crear el botón Ejercicio 2 en la nueva ventana
    ejercicio2_button = tk.Button(
        segunda_ventana,
        text="Ejercicio 2",
        command=crear_cuarta_ventana,  # Reemplazar con tu acción para Ejercicio 2
    )
    ejercicio2_button.pack(pady=5)


# Función para crear una nueva ventana cuando se presiona Ejercicio 1
def crear_tercera_ventana():
    tercera_ventana = tk.Toplevel()
    tercera_ventana.title("Tercera Ventana")
    tercera_ventana.configure(bg="black")

    # Agregar etiqueta de instrucción
    etiqueta_instruccion = tk.Label(
        tercera_ventana,
        text="Escoger función a girar y mover",
        font=("Helvetica", 16),
        fg="white",
        bg="black",
    )
    etiqueta_instruccion.pack(pady=10)

    def x_n_accion():
        try:
            # Ejecutar el archivo "xn.py" sin especificar una ruta
            subprocess.run(["python", "xn.py"])
        except FileNotFoundError:
            print("Error: No se encontró el archivo 'xn.py'.")

    # Crear dos botones "x[n]" y "h[n]"
    x_n_button = tk.Button(
        tercera_ventana,
        text="x[n]",
        command=x_n_accion,  # Reemplazar con tu acción para x[n]
    )
    x_n_button.pack(pady=5)

    # Función que se llama cuando se hace clic en el botón "h[n]"
    def h_n_accion():
        try:
            # Ejecutar el archivo "hn.py" sin especificar una ruta
            subprocess.run(["python", "hn.py"])
        except FileNotFoundError:
            print("Error: No se encontró el archivo 'hn.py'.")

    h_n_button = tk.Button(
        tercera_ventana,
        text="h[n]",
        command=h_n_accion,
    )
    h_n_button.pack(pady=5)


# Función para crear una nueva ventana cuando se presiona Ejercicio 2
def crear_cuarta_ventana():
    cuarta_ventana = tk.Toplevel()
    cuarta_ventana.title("Cuarta Ventana")
    cuarta_ventana.configure(bg="black")

    # Agregar etiqueta de instrucción
    etiqueta_instruccion = tk.Label(
        cuarta_ventana,
        text="Escoger función a girar y mover",
        font=("Helvetica", 16),
        fg="white",
        bg="black",
    )
    etiqueta_instruccion.pack(pady=10)

    def x_n_accion():
        try:
            # Ejecutar el archivo "xn2.py" sin especificar una ruta
            subprocess.run(["python", "xn2.py"])
        except FileNotFoundError:
            print("Error: No se encontró el archivo 'xn2.py'.")

    # Crear dos botones "x[n]" y "h[n]"
    x_n_button = tk.Button(
        cuarta_ventana,
        text="x[n]",
        command=x_n_accion,  # Reemplazar con tu acción para x[n]
    )
    x_n_button.pack(pady=5)

    # Función que se llama cuando se hace clic en el botón "h[n]"
    def h_n_accion():
        try:
            # Ejecutar el archivo "hn2.py" sin especificar una ruta
            subprocess.run(["python", "hn2.py"])
        except FileNotFoundError:
            print("Error: No se encontró el archivo 'hn2.py'.")

    h_n_button = tk.Button(
        cuarta_ventana,
        text="h[n]",
        command=h_n_accion,
    )
    h_n_button.pack(pady=5)


# Función que se llama cuando se hace clic en el botón Tiempo Discreto
def accion_tiempo_discreto():
    # Crear una nueva ventana con botones de Ejercicio 1 y Ejercicio 2
    crear_segunda_ventana()


# Función que se llama cuando se hace clic en el botón Tiempo Continuo
def accion_tiempo_continuo():
    # Crear una nueva ventana para ingresar valores
    ventana_input = tk.Toplevel()
    ventana_input.title("Valores de Entrada")
    ventana_input.configure(bg="black")

    # Crear etiquetas para los campos de entrada
    etiqueta_a = tk.Label(
        ventana_input,
        text="Seleccionar la primera señal '(a-f)':",
        font=("Helvetica", 16),
        fg="white",
        bg="black",
    )
    etiqueta_a.pack(pady=10)

    # Crear un widget Entry para 'a'
    entrada_a = Entry(ventana_input)
    entrada_a.pack(pady=5)

    etiqueta_b = tk.Label(
        ventana_input,
        text="Seleccionar la segunda señal '(a-f)':",
        font=("Helvetica", 16),
        fg="white",
        bg="black",
    )
    etiqueta_b.pack(pady=10)

    # Crear un widget Entry para 'b'
    entrada_b = Entry(ventana_input)
    entrada_b.pack(pady=5)

    # Crear una función para ejecutar el script con los valores proporcionados
    def ejecutar_script():
        valor_a = entrada_a.get()  # Obtener el valor de 'a' del widget Entry
        valor_b = entrada_b.get()  # Obtener el valor de 'b' del widget Entry

        # Usar el método Popen con stdin para enviar las entradas al script
        try:
            proceso = subprocess.Popen(
                ["python", "continuas_animadas.py"], stdin=subprocess.PIPE
            )
            proceso.communicate(input=f"{valor_a}\n{valor_b}\n".encode())
        except FileNotFoundError:
            print("Error: Se produjo un error al procesar las señales.")

        # Cerrar la ventana de entrada
        ventana_input.destroy()

    # Crear un botón para ejecutar el script con los valores proporcionados
    boton_ejecutar = tk.Button(ventana_input, text="Ejecutar", command=ejecutar_script)
    boton_ejecutar.pack(pady=10)


# Crear una ventana principal
root = tk.Tk()
root.title("Bernal Garcia Sanchez")
root.configure(bg="black")

# Crear una etiqueta con tu nombre
etiqueta = tk.Label(
    root,
    text="Laboratorio de Convolución",
    font=("Helvetica", 24),
    fg="white",
    bg="black",
)
etiqueta.grid(row=0, pady=10)

# Centrar la etiqueta
etiqueta.grid_configure(sticky="nsew")

# Crear el botón Tiempo Discreto
boton_tiempo_discreto = tk.Button(
    root,
    text="Tiempo Discreto",
    command=accion_tiempo_discreto,  # Definir la acción para este botón
)
boton_tiempo_discreto.grid(row=1, pady=5)

# Crear el botón Tiempo Continuo
boton_tiempo_continuo = tk.Button(
    root,
    text="Tiempo Continuo",
    command=accion_tiempo_continuo,  # Definir la acción para este botón
)
boton_tiempo_continuo.grid(row=2, pady=5)

# Crear una etiqueta para los nombres
etiqueta_nombres = tk.Label(
    root,
    text="Cristian Bernal, José García y Juan Sánchez",
    font=("Helvetica", 18),
    fg="white",
    bg="black",
)
etiqueta_nombres.grid(row=3, pady=10)

# Configurar filas y columnas de la cuadrícula para expandirse
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Iniciar la aplicación GUI
root.geometry("1600x900")
root.mainloop()
