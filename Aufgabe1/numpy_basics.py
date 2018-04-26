import numpy as np
import math

# a) (*) Erzeugen Sie einen Vektor mit Nullen der Länge 10 (10 Elemente) und setzen den Wert des
#        5.Elementes auf eine 1.
array = np.zeros(10)
array[4] = 1
print("Aufgabe a:")
print(array)
print("#######################################################")
# b) (*) Erzeugen Sie einen Vektor mit Ganzahl-Werten von 10 bis 49 (geht in einer Zeile).
vec = np.arange(10, 50, 1)
print("Aufgabe b:")
print(vec)
print("#######################################################")
# c) (*) Drehen Sie die Werte des Vektors um (geht in einer Zeile).
print("Aufgabe c:")
print(vec[::-1])
print("#######################################################")
# d) (*) Erzeugen Sie eine 4x4 Matrix mit den Werte 0 bis 15 (links oben rechts unten).
print("Aufgabe d:")
print(np.arange(16).reshape(4, 4))
print("#######################################################")
# e) (*) Erzeuge eine 8x8 Matrix mit Zufallswerte und finde deren Maximum und Minimum und
#        normalisieren Sie die Werte (sodass alle Werte zwischen 0 und 1 liegen - ein Wert wird 1 (max)
#        sein und einer 0 (min)).
matrix = np.random.random((8, 8))
print("Aufgabe e Zufallsmatrix:")
print(matrix)
matrixmax, matrixmin = matrix.max(), matrix.min()
matrix = (matrix - matrixmin) / (matrixmax - matrixmin)
print("Aufgabe e normalisierte Matrix:")
print(matrix)
print("#######################################################")
# f) (*) Multiplizieren Sie eine 4x3 Matrix mit einer 3x2 Matrix
print(np.ones((3, 2)))
print(np.ones((4, 3)))
mult = np.dot(np.ones((4, 3)), np.ones((3, 2)))
print("Aufgabe f:")
print(mult)
print("#######################################################")
# g) (*) Erzeugen Sie ein 1D Array mit den Werte von 0 bis 20 und negieren Sie Werte zwischen 8
#        und 16 nachträglich.
arrayNegate = np.arange(0, 21, 1)
for i in arrayNegate[9: 16]:
    arrayNegate[i] = i * -1
print("Aufgabe g:")
print(arrayNegate)
print("#######################################################")
# h) (*) Summieren Sie alle Werte in einem Array.
print("Aufgabe h (Summe von g. )")
print(np.sum(arrayNegate))
print("#######################################################")
# i) (** ) Erzeugen Sie eine 5x5 Matrix und geben Sie jeweils die geraden und die ungeraden Zeile
#          aus.
matrixE = np.arange(25).reshape(5, 5)
print(matrixE)
print("Aufgabe i Gerade: ")
# startAt:endBefore:skip
print(matrixE[::2])
print("Aufgabe i Ungerade: ")
print(matrixE[1::2])
print("#######################################################")
# x = np.arange(0, 21, 1)
# print(x)
# print(x[0::2])
# print(x[1::2])

# j) (** ) Erzeugen Sie eine Matrix M der Größe 4x3 und einen Vektor v mit Länge 3. Multiplizieren
#          Sie jeden Spalteneintrag aus v mit der kompletten Spalte aus M. Schauen Sie sich dafür an, was
#          Broadcasting in Numpy bedeutet.
print("Aufgabe j nicht gelöst")
print("#######################################################")
# k) (** ) Erzeugen Sie einen Zufallsmatrix der Größe 10x2, die Sie als Kartesische Koordinaten
#          interpretieren können ([[x0, y0],[x1, y1],[x2, y2]]). Konvertieren Sie diese in Polarkoordinaten
#          https://de.wikipedia.org/wiki/Polarkoordinaten.
matrixR = np.random.random((10, 2))
x, y = matrixR[:, 0], matrixR[:, 1]
# Radius
r = np.sqrt(y ** 2 + y ** 2)
# Einige Programmiersprachen und Anwendungsprogramme (etwa Microsoft Excel) bieten eine Arkustangens-Funktion
# (y,x) mit zwei Argumenten an, welche die dargestellten Fallunterscheidungen intern berücksichtigt und den korrekten
# Wert für \varphi  für beliebige Werte von x und y berechnet.
t = np.arctan2(y, x)

print("Aufgabe k Koordinaten:")
print(r)
print("Aufgabe k Polarkoordinaten:")
print(t)
print("#######################################################")


# l) (***) Implementieren Sie zwei Funktionen, die das Skalarprodukt und die Vektorlänge für Vektoren
#       beliebiger Länge berechnen. Nutzen Sie dabei NICHT die gegebenen Funktionen von
#       NumPy. Testen Sie Ihre Funktionen mit den gegebenen Vektoren:

# Vektorlänge: (a,b) : sqrt(a^2+b^2)
def vector_length(vector):
    summe = 0.0
    for i in vector:
        summe = i ** 2 + summe
    # np.sqrt ?
    return math.sqrt(summe)


print("Aufgabe l Vektorlänge (2,3,1):")
print(vector_length(np.array([2, 3, 1])))   # 3,7416...
print("#######################################################")


def scalar_product(vector_a, vector_b):
    scalar = 0.0
    for j, item in enumerate(vector_a):
        scalar = vector_a[j] * vector_b[j] + scalar
    return scalar


print("Aufgabe l Scalar (2,-5,0),(3,2,5):")
print(scalar_product(np.array([2, -4, 0]), np.array([3, 2, 5])))  # -2

print("#######################################################")
print("Aufgabe m nicht gelöst")
