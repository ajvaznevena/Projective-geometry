import sys
import math
import numpy as np
from numpy import linalg as la

from tkinter import *
from PIL import ImageTk, Image


class Projekcija:

    def __init__(self, root, frame):

        self.root = root
        self.frame = frame
        self.niz_tacaka = []
        self.niz_tacaka1 = []
        self.niz = []       # za dlt i modifikovani dlt
        self.niz1 = []

        # Donja polovina prozora - okvir za dugmice
        self.bottomFrame = Frame(self.root)
        self.bottomFrame.pack(side=BOTTOM, expand=1)
        #1
        self.unesi_koordinate = Button(self.bottomFrame, text="Unesi koordinate", command=self.unesi_tacke)
        self.unesi_koordinate.pack(side=LEFT, padx=0, pady=10)
        #2
        self.ucitaj_sliku = Button(self.bottomFrame, text="Ucitaj fotografiju", command=self.ucitaj_sliku)
        self.ucitaj_sliku.pack(side=LEFT, padx=10, pady=10)
        #3
        self.ispravi_sliku = Button(self.bottomFrame, text="Ispravi sliku", command=self.napravi_pravougaonik)
        self.ispravi_sliku.pack(side=LEFT, padx=10, pady=10)


    def ucitaj_sliku(self):

        # LEVA SLIKA
        self.fotografija = ImageTk.PhotoImage(Image.open("building.jpg"))
        height = self.fotografija.height()
        width = self.fotografija.width()

        # Levi kanvas
        self.canvas_levo = Canvas(self.root, width=width, height=height)
        self.canvas_levo.place(x=100, y=120)
        self.canvas_levo.bind('<Button-1>', self.napravi_tacku)

        self.canvas_levo.create_image(0,0, image=self.fotografija, anchor="nw")

        # DESNA SLIKA
        self.canvas_desno = Canvas(self.root, width=width, height=height, background="black")
        self.canvas_desno.place(x=450, y=120)
        self.canvas_desno.bind('<Button-1>', self.napravi_tacku1)


    def napravi_tacku(self, event):
        x, y = event.x, event.y
        print("Koordinata x: " + str(x))
        print("Koordinata y: " + str(y))
        print("-------------")

        # Crta se crveni kruzic
        self.canvas_levo.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="black", width=1)

        # Dodaje kliknutu tacku u niz tacaka
        self.niz_tacaka.append((x,y))

    def napravi_tacku1(self, event):
        x, y = event.x, event.y
        print("Koordinata x: " + str(x))
        print("Koordinata y: " + str(y))
        print("-------------")

        # Crta se plavi kruzic
        self.canvas_desno.create_oval(x-5, y-5, x+5, y+5, fill="blue", outline="white", width=1)

        # Dodaje kliknutu tacku u niz tacaka
        self.niz_tacaka1.append((x,y))

    def napravi_pravougaonik(self):

        for tacka in self.niz_tacaka:
            print(tacka)

        self.img = Image.open("building.jpg")

        width, height = self.img.size

        P1 = self.naivni(self.niz_tacaka[0][0], self.niz_tacaka[0][1], self.niz_tacaka[1][0],
                        self.niz_tacaka[1][1], self.niz_tacaka[2][0], self.niz_tacaka[2][1],
                        self.niz_tacaka[3][0], self.niz_tacaka[3][1], self.niz_tacaka1[0][0],
                        self.niz_tacaka1[0][1],self.niz_tacaka1[1][0],self.niz_tacaka1[1][1],
                        self.niz_tacaka1[2][0],self.niz_tacaka1[2][1],self.niz_tacaka1[3][0],
                        self.niz_tacaka1[3][1]
                        )

        P2 = self.dlt(self.niz_tacaka[0][0], self.niz_tacaka[0][1], self.niz_tacaka[1][0],
                        self.niz_tacaka[1][1], self.niz_tacaka[2][0], self.niz_tacaka[2][1],
                        self.niz_tacaka[3][0], self.niz_tacaka[3][1], self.niz_tacaka1[0][0],
                        self.niz_tacaka1[0][1], self.niz_tacaka1[1][0], self.niz_tacaka1[1][1],
                        self.niz_tacaka1[2][0], self.niz_tacaka1[2][1], self.niz_tacaka1[3][0],
                        self.niz_tacaka1[3][1]
                        )

        P3 = self.modifikovani_dlt(self.niz_tacaka[0][0], self.niz_tacaka[0][1], self.niz_tacaka[1][0],
                      self.niz_tacaka[1][1], self.niz_tacaka[2][0], self.niz_tacaka[2][1],
                      self.niz_tacaka[3][0], self.niz_tacaka[3][1], self.niz_tacaka1[0][0],
                      self.niz_tacaka1[0][1], self.niz_tacaka1[1][0], self.niz_tacaka1[1][1],
                      self.niz_tacaka1[2][0], self.niz_tacaka1[2][1], self.niz_tacaka1[3][0],
                      self.niz_tacaka1[3][1]
                      )

        self.kopija = Image.new("RGB", (width, height), "black")


        P1_invertovano = la.inv(P1)     #naivni
        P2_invertovano = la.inv(P2)     #dlt
        P3_invertovano = la.inv(P3)     #modifikovani_dlt


        cols = self.kopija.size[0]
        rows = self.kopija.size[1]

        for i in range(cols):
            for j in range(rows):
                nove_koordinate = P3_invertovano.dot([i, j, 1])
                nove_koordinate = [(x / nove_koordinate[2]) for x in nove_koordinate]

                if (nove_koordinate[0] >= 0 and nove_koordinate[0] < cols - 1 and nove_koordinate[1] >= 0 and nove_koordinate[1] < rows - 1):
                    tmp1 = self.img.getpixel((math.floor(nove_koordinate[0]), math.floor(nove_koordinate[1])))
                    tmp2 = self.img.getpixel((math.ceil(nove_koordinate[0]), math.ceil(nove_koordinate[1])))
                    self.kopija.putpixel((i, j), tmp2)

        self.kopija.save("izlaz.jpg")

        self.labela = Label(self.root, width=width, height=height)
        self.labela.place(x=451, y=121)

        self.izlaz = ImageTk.PhotoImage(Image.open("izlaz.jpg"))
        self.labela.config(image=self.izlaz)


    def unesi_tacke(self):

        print("UNOSENJE KOORDINATA PRVOG PRAVOUGAONIKA...")

        x1, y1 = [float(c) for c in input("Unesite koordinate tacke A: ").split()]
        x2, y2 = [float(c) for c in input("Unesite koordinate tacke B: ").split()]
        x3, y3 = [float(c) for c in input("Unesite koordinate tacke C: ").split()]
        x4, y4 = [float(c) for c in input("Unesite koordinate tacke D: ").split()]

        print("\nUNOSENJE KOORDINATA DRUGOG PRAVOUGAONIKA...")

        x1p, y1p = [float(c) for c in input("Unesite koordinate tacke A: ").split()]
        x2p, y2p = [float(c) for c in input("Unesite koordinate tacke B: ").split()]
        x3p, y3p = [float(c) for c in input("Unesite koordinate tacke C: ").split()]
        x4p, y4p = [float(c) for c in input("Unesite koordinate tacke D: ").split()]

        self.naivni(x1, y1, x2, y2, x3, y3, x4, y4,   x1p, y1p, x2p, y2p, x3p, y3p, x4p, y4p)

        #self.dlt(x1, y1, x2, y2, x3, y3, x4, y4,  x1p, y1p, x2p, y2p, x3p, y3p, x4p, y4p)
        #self.modifikovani_dlt(x1, y1, x2, y2, x3, y3, x4, y4,  x1p, y1p, x2p, y2p, x3p, y3p, x4p, y4p)

        a = input("UNESITE PRVE TACKE ZA DLT I MODIFIKOVANI DLT ILI STOP ZA KRAJ: ")
        while a != ("stop"):
            x, y  = a.split(" ")
            self.niz.append((float(x),float(y)))
            a = input("UNESITE PRVE TACKE ZA DLT I MODIFIKOVANI DLT ILI STOP ZA KRAJ: ")


        a = input("\nUNESITE DRUGE TACKE ZA DLT I MODIFIKOVANI DLT ILI STOP ZA KRAJ: ")
        while a != ("stop"):
            x, y  = a.split(" ")
            self.niz1.append((float(x),float(y)))
            a = input("UNESITE RUGE TACKE ZA DLT I MODIFIKOVANI DLT ILI STOP ZA KRAJ: ")


        self.dlt_niz_tacaka(self.niz, self.niz1)
        self.modifikovani_dlt_niz_tacaka(self.niz, self.niz1)


    def matdet(self, x1, x2, x3, y1, y2, y3):
        matrica = np.array([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]], dtype='float')
        delta = la.det(matrica)
        return matrica, delta


    def naivni(self, x1, y1, x2, y2, x3, y3, x4, y4,  x1p, y1p, x2p, y2p, x3p, y3p, x4p, y4p):

        # Naivni algoritam A -> A', B->B', C->C', D->D', 3x3 matrica kao izlaz
        # A, B, C, D  => A, B, C kolinearne
        # D = l1*A + l2*B + l3*C, gde nijedno l ne sme biti 0

        # Resavanje sistema radimo Kramerovim pravilom:
        matrica, delta = self.matdet(x1, x2, x3, y1, y2, y3)
        matrica1, delta1 = self.matdet(x4, x2, x3, y4, y2, y3)
        matrica2, delta2 = self.matdet(x1, x4, x3, y1, y4, y3)
        matrica3, delta3 = self.matdet(x1, x2, x4, y1, y2, y4)

        l1 = delta1 / delta
        l2 = delta2 / delta
        l3 = delta3 / delta

        if ((l1 or l2 or l3) == 0):
            sys.stderr("Vrednost lambde ne sme biti nula!")

        P1 = np.array([[l1 * x1, l2 * x2, l3 * x3],
                       [l1 * y1, l2 * y2, l3 * y3],
                       [l1, l2, l3]])

        P1_inverz = la.inv(P1)

        # imam P1, ostaje nam P2 da bismo izracunali P2*P1^-1

        matricap, deltap = self.matdet(x1p, x2p, x3p, y1p, y2p, y3p)
        matrica1p, delta1p = self.matdet(x4p, x2p, x3p, y4p, y2p, y3p)
        matrica2p, delta2p = self.matdet(x1p, x4p, x3p, y1p, y4p, y3p)
        matrica3p, delta3p = self.matdet(x1p, x2p, x4p, y1p, y2p, y4p)

        l1p = delta1p / deltap
        l2p = delta2p / deltap
        l3p = delta3p / deltap

        if ((l1p or l2p or l3p) == 0):
            sys.stderr("Vrednost lambde ne sme biti nula!")

        P2 = np.array([[l1p * x1p, l2p * x2p, l3p * x3p],
                       [l1p * y1p, l2p * y2p, l3p * y3p],
                       [l1p, l2p, l3p]])

        # rezultat je P:

        P = np.dot(P2, P1_inverz)

        print("\nKONACNO RESENJE NAIVNI JE:")
        for row in P:
            print(np.around(row, decimals=6))

        return P

    def napravi_2x9_matricu(self, x1, x2, y1, y2):

        # x1 = x koordinata tacke A
        # x2 = y koordinata tacke A
        # y1 = x koordinata tacke Ap
        # y2 = y koordinata tacke Ap

        # Pravimo matricu 2x9
        a = np.zeros(shape=(2, 9))

        # Racunaju se vrednosti u prvom redu
        a[0][0] = 0;        a[0][1] = 0;        a[0][2] = 0;
        a[0][3] = -1 * x1;  a[0][4] = -1 * x2;  a[0][5] = -1 * 1;
        a[0][6] = y2 * x1;  a[0][7] = y2 * x2;  a[0][8] = y2 * 1;

        # Racunaju se vrednosti u drugom redu
        a[1][0] = 1 * x1;   a[1][1] = 1 * x2;   a[1][2] = 1 * 1;
        a[1][3] = 0;        a[1][4] = 0;        a[1][5] = 0;
        a[1][6] = -y1 * x1; a[1][7] = -y1 * x2; a[1][8] = -y1 * 1;

        return a


    def dlt(self, x1, y1, x2, y2, x3, y3, x4, y4,  x1p, y1p, x2p, y2p, x3p, y3p, x4p, y4p):

        #odrediti 2x9 matricu.

        a1 = self.napravi_2x9_matricu(x1, y1, x1p, y1p)
        a2 = self.napravi_2x9_matricu(x2, y2, x2p, y2p)
        a3 = self.napravi_2x9_matricu(x3, y3, x3p, y3p)
        a4 = self.napravi_2x9_matricu(x4, y4, x4p, y4p)

        # 2. KORAK:
        # Spojiti te matrice u jednu matricu A formata 2n x 9.
        n = 4
        A = np.zeros(shape=(2 * n, 9))

        # Dodavanje izracunatih vrednosti na rezultatsku matricu
        A[0] = a1[0]
        A[1] = a1[1]
        A[2] = a2[0]
        A[3] = a2[1]
        A[4] = a3[0]
        A[5] = a3[1]
        A[6] = a4[0]
        A[7] = a4[1]

        U, s, V = np.linalg.svd(A, full_matrices=True)
        A1 = V[8].reshape(3, 3)

        print("KONACNO RESENJE DLT: ")

        for row in A1:
            print(np.around(row, decimals=6))

        return A1

    def dlt_niz_tacaka(self, niz, niz1):

        a = [0 for i in range(len(niz))]

        for i in range(len(niz)):
            a[i] = self.napravi_2x9_matricu(niz[i][0], niz[i][1], niz1[i][0], niz1[i][1])

        n = len(niz)
        A = np.zeros(shape=(2 * n, 9))

        for i in range(len(niz)):
            A[2*i] = a[i][0]
            A[2*i+1] = a[i][1]

        U, s, V = np.linalg.svd(A, full_matrices=True)
        A1 = V[8].reshape(3, 3)

        print("\n\nKONACNO RESENJE DLT ZA PROIZVOLJAN BROJ TACAKA\n")
        for row in A1:
            print(np.around(row, decimals=5))


    def normalizuj_tacke(self, A):

        # 1. korak
        # Izracunati teziste sistema tacaka (afino)

        Tx = sum(list(map(lambda x: x[0], A))) / len(A)
        Ty = sum(list(map(lambda x: x[1], A))) / len(A)

        # T = np.array([Tx, Ty, 1])
        # print(T)

        # 2. korak
        # Translirati teziste u koordinatni pocetak (matrica translacije G)

        G = np.array([
            [1, 0, -Tx],
            [0, 1, -Ty],
            [0, 0, 1]
        ])
        # print(G)

        # Primenjuje se matrica G na svaku tacku
        M_trans = list(map(lambda x: np.dot(G, x), A))

        # 3. korak
        # Skalirati tacke tako da prosecna udaljenost tacke od koordinatnog pocetka
        # bude sqrt(2) (matrica homotetije S)

        # Racuna se prosecno rastojanje od koordinatnog pocetka
        avg = sum(list(map(lambda x: math.sqrt(x[0] ** 2 + x[1] ** 2), M_trans))) / len(M_trans)

        # Skaliranje tacaka, udaljenost treba da bude sqrt(2)
        S = np.array([
            [math.sqrt(2) / avg, 0, 0],
            [0, math.sqrt(2) / avg, 0],
            [0, 0, 1]
        ])

        # 4. korak
        # Matrica normalizacije T = S*G
        T = np.dot(S, G)

        return T

    def dltM(self, M, Mp):

        # Raspakivanje

        x1 = M[0][0]
        y1 = M[0][1]
        x2 = M[1][0]
        y2 = M[1][1]
        x3 = M[2][0]
        y3 = M[2][1]
        x4 = M[3][0]
        y4 = M[3][1]

        x1p = Mp[0][0]
        y1p = Mp[0][1]
        x2p = Mp[1][0]
        y2p = Mp[1][1]
        x3p = Mp[2][0]
        y3p = Mp[2][1]
        x4p = Mp[3][0]
        y4p = Mp[3][1]

        a1 = self.napravi_2x9_matricu(x1, y1, x1p, y1p)
        a2 = self.napravi_2x9_matricu(x2, y2, x2p, y2p)
        a3 = self.napravi_2x9_matricu(x3, y3, x3p, y3p)
        a4 = self.napravi_2x9_matricu(x4, y4, x4p, y4p)

        # Spojiti te matrice u jednu matricu A formata 2n x 9.
        n = 4
        A = np.zeros(shape=(2 * n, 9))

        # Dodavanje izracunatih vrednosti na rezultatsku matricu
        A[0] = a1[0]
        A[1] = a1[1]
        A[2] = a2[0]
        A[3] = a2[1]
        A[4] = a3[0]
        A[5] = a3[1]
        A[6] = a4[0]
        A[7] = a4[1]

        U, s, V = np.linalg.svd(A, full_matrices=True)
        A1 = V[8].reshape(3, 3)

        return A1

    def modifikovani_dlt_niz_tacaka(self, niz, niz1):

        P1 = np.ones((len(niz), 3))
        for i in range(len(niz)):
            P1[i][0] = niz[i][0]
            P1[i][1] = niz[i][1]


        P2 = np.ones((len(niz1), 3))
        for i in range(len(niz1)):
            P2[i][0] = niz1[i][0]
            P2[i][1] = niz1[i][1]

        # Prvo se dobija transformacija T
        T = self.normalizuj_tacke(P1)

        # A sada se normalizuju originalne tacke
        P1_norm = np.array(list(map(lambda x: np.dot(T, x), P1)))

        # Prvo se dobija transformacija T'
        Tp = self.normalizuj_tacke(P2)
        # print(Tp)

        # A sada se normalizuju slike tacaka
        P2_norm = np.array(list(map(lambda x: np.dot(Tp, x), P2)))

        # dlt algoritmom matrica transformacije P̄ iz korespodencija P1_norm, P2_norm

        P_norm = self.dltM(P1_norm, P2_norm)

        # Trazena matrica transformacije je P = T'^(-1) * P * T

        Tp = np.linalg.inv(Tp)
        P = np.dot(np.dot(Tp, P_norm), T)

        print("\n\nKONACNO RESENJE ZA MODIFIKOVANI SA PROIZVOLJNIM BROJEM TACAKA JE:\n")

        for row in P:
          print(np.around(row, decimals=5))

        return P


    def modifikovani_dlt(self, x1, y1, x2, y2, x3, y3, x4, y4, x1p, y1p, x2p, y2p, x3p, y3p, x4p, y4p):

        P1 = np.array([[x1, y1, 1],
                       [x2, y2, 1],
                       [x3, y3, 1],
                       [x4, y4, 1]])

        P2 = np.array([[x1p, y1p, 1],
                       [x2p, y2p, 1],
                       [x3p, y3p, 1],
                       [x4p, y4p, 1]])


        # Prvo se dobija transformacija T
        T = self.normalizuj_tacke(P1)

        # A sada se normalizuju originalne tacke
        P1_norm = np.array(list(map(lambda x: np.dot(T, x), P1)))

        # Prvo se dobija transformacija T'
        Tp = self.normalizuj_tacke(P2)

        # A sada se normalizuju slike tacaka
        P2_norm = np.array(list(map(lambda x: np.dot(Tp, x), P2)))

        # dlt algoritmom matrica transformacije P̄ iz korespodencija P1_norm, P2_norm

        P_norm = self.dltM(P1_norm, P2_norm)

        # Trazena matrica transformacije je P = T'^(-1) * P * T

        Tp = np.linalg.inv(Tp)
        P = np.dot(np.dot(Tp, P_norm), T)

        print("KONACNO RESENJE ZA MODIFIKOVANI JE:")

        for row in P:
            print(np.around(row, decimals=6))

        return P

def main():
    root = Tk()
    root.title("Projektivna geometrija")
    root.resizable(False, False)

    frame = Frame(root, width=800, height=600)
    frame.pack_propagate(0)
    frame.pack(fill=BOTH, expand=1)


    n = Projekcija(root, frame)

    root.mainloop()

    sys.exit(0)

if __name__ == '__main__':
    main()
