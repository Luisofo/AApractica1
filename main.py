###############################################################################
#TRABAJO 1.
#Nombre Estudiante: Luis Fernandez Garcia
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

##################################EJERCICIO 1##################################


def E(u, v):
    return (u * v * np.e**(-u**2-v**2))**2


# Derivada parcial de E con respecto a u
def dEu(u, v):
    return -2.0*u*(2.0*u**2-1)*v**2*np.e**(-2.0*(u**2+v**2))


# Derivada parcial de E con respecto a v
def dEv(u, v):
   return  -2.0*v*(2.0*v**2-1)*u**2*np.e**(-2.0*(u**2+v**2))


# Gradiente de E
def gradE(u, v):
   return np.array([dEu(u, v), dEv(u, v)])


def F(x, y):
   return x ** 2 + 2 * y ** 2 + 2 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)


# Derivada parcial de F con respecto a x
def dFx(x, y):
   return 2*(2*np.pi*np.cos(2*np.pi*x)*np.sin(np.pi*y)+x)


# Derivada parcial de F con respecto a y
def dFy(x, y):
   return 2*np.pi*np.sin(2*np.pi*x)*np.cos(np.pi*y)+4*y


# Gradiente de E
def gradF(x, y):
   return np.array([dFx(x, y), dFy(x, y)])


def gradient_descent(lr, grad_fun, fun, epsilon, max_iters, w_ini):

   valores = []                              # Lista donde añadiremos los valores que calculemos iterativamente
   w = w_ini                                 # Igualamos la w a la w_inicial y
   it = 0                                    # la añadimos a la lista junto su indice (it = 0)
   valor_f = fun(w[0], w[1])
   valores.append([it, valor_f,w_ini])

   while it<max_iters and valor_f>epsilon:
      it = it+1
      w = w - lr*grad_fun(w[0],w[1])
      valor_f = fun(w[0],w[1])
      valores.append([it,valor_f,[w[0],w[1]]])

   return valores,w,valor_f


###############################################################################
#Esta función muestra una figura 3D con la función a optimizar junto con el
#óptimo encontrado y la ruta seguida durante la optimización. Esta función, al igual
#que las otras incluidas en este documento, sirven solamente como referencia y
#apoyo a los estudiantes. No es obligatorio emplearlas, y pueden ser modificadas
#como se prefiera.
#    rng_val: rango de valores a muestrear en np.linspace()
#    fun: función a optimizar y mostrar
#    ws: conjunto de pesos (pares de valores [x,y] que va recorriendo el optimizador
#                           en su búsqueda iterativa del óptimo)
#    colormap: mapa de color empleado en la visualización
#    title_fig: título superior de la figura
#
#Ejemplo de uso: display_figure(2, E, ws, 'plasma','Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
###############################################################################


def display_figure(rng_val, fun, ws, colormap, title_fig):
   # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
   from mpl_toolkits.mplot3d import Axes3D
   E2 = fun
   x = np.linspace(-rng_val, rng_val, 50)
   y = np.linspace(-rng_val, rng_val, 50)
   X, Y = np.meshgrid(x, y)
   Z = fun(X, Y)
   fig = plt.figure()
   ax = Axes3D(fig, auto_add_to_figure=False)
   fig.add_axes(ax)
   ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                   cstride=1, cmap=colormap, alpha=.6)
   if len(ws) > 0:
      ws = np.asarray(ws)
      min_point = np.array([ws[-1, 0], ws[-1, 1]])
      min_point_ = min_point[:, np.newaxis]
      ax.plot(ws[:-1, 0], ws[:-1, 1], E2(ws[:-1, 0], ws[:-1, 1]), 'r*', markersize=5)
      ax.plot(min_point_[0], min_point_[1], E2(min_point_[0], min_point_[1]), 'r*', markersize=10)
   if len(title_fig) > 0:
      fig.suptitle(title_fig, fontsize=16)
   ax.set_xlabel('u')
   ax.set_ylabel('v')
   ax.set_zlabel('E(u,v)')


def ejercicio1():

   print("Ejercicio 1.2")
   learning_rate = 0.1
   error = 1e-8
   iteraciones_max = 100000
   w = [0.5,-0.5]

   valores, w, fun = gradient_descent(learning_rate, gradE, E, error, iteraciones_max, w)

   iteraciones = [x[0] for x in valores]
   valores_F = [x[1] for x in valores]
   pesos = [x[2] for x in valores]
   display_figure(3, E, pesos, colormap='viridis', title_fig="KLK")

   print("iteracion = ", valores[-1][0],"pesos = ", w,"valor de la funcion = ", fun)

   ############################################################################

   print("Ejercicio 1.3")
   learning_rate = 0.01
   error = -100
   iteraciones_max = 50
   w = [-1, 1]

   valores, w, fun = gradient_descent(learning_rate, gradF, F, error, iteraciones_max, w)

   iteraciones= [x[0] for x in valores]
   valores_F = [x[1] for x in valores]
   pesos = [x[2] for x in valores]
   print(iteraciones)
   print(valores_F)
   plt.plot(iteraciones,valores_F)
   plt.show()

   learning_rate = 0.1
   error = -100
   iteraciones_max = 50
   w = [-1, 1]

   valores, w, fun = gradient_descent(learning_rate, gradF, F, error, iteraciones_max, w)

   iteraciones = [x[0] for x in valores]
   valores_F = [x[1] for x in valores]
   print(iteraciones)
   print(valores_F)
   plt.plot(iteraciones, valores_F)
   plt.show()

   valores_iniciales = [[-0.5,-0.5],[1,1],[2.1,-2.1],[-3,3],[-2,2]]
   learning_rate = [0.01,0.1]
   tabla_valores = []

   ############################################################################

   print("Ejercicio 1.4")

   for x in valores_iniciales:
      for y in learning_rate:
         valores, w, fun = gradient_descent(y, gradF, F, error, iteraciones_max, x)

         iteraciones = [valor[0] for valor in valores]
         valores_F = [valor[1] for valor in valores]
         tabla_valores.append([y,x,fun,w])
         plt.plot(iteraciones, valores_F)
         titulo = "Valores iniciales = (" + str(x[0]) +", " +str(x[1]) + ") Learning rate = " + str(y)
         plt.suptitle(titulo)
         plt.show()

   print(tabla_valores)


##################################EJERCICIO 2##################################
label5 = 1
label1 = -1


# Funcion para leer los datos
def readData(file_x, file_y):
   # Leemos los ficheros
   datax = np.load(file_x)
   datay = np.load(file_y)
   y = []
   x = []
   # Solo guardamos los datos cuya clase sea la 1 o la 5
   for i in range(0, datay.size):
      if datay[i] == 5 or datay[i] == 1:
         if datay[i] == 5:
            y.append(label5)
         else:
            y.append(label1)
         x.append(np.array([1, datax[i][0], datax[i][1]]))

   x = np.array(x, np.float64)
   y = np.array(y, np.float64)

   return x, y


# Funcion para calcular el error
def Err(x, y, w):
   return


# Gradiente Descendente Estocastico
def sgd(algo):
   #
   return w


# Pseudoinversa
def pseudoinverse(algo):
   #
   return w


# Lectura de los datos de entrenamiento
# x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
# x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

#w = sgd('?')
#print('Bondad del resultado para grad. descendente estocastico:\n')
#print("Ein: ", Err(x, y, w))
#print("Eout: ", Err(x_test, y_test, w))

#input("\n--- Pulsar tecla para continuar ---\n")

# Seguir haciendo el ejercicio...

#print('Ejercicio 2\n')


# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
   return np.random.uniform(-size, size, (N, d))


def sign(x):
   if x >= 0:
      return 1
   return -1


def f(x1, x2):
   return sign((x1 - 0.2) ** 2 + x2 ** 2 - 0.6)


if __name__ == '__main__':
   ejercicio1()

