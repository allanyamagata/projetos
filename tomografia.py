# MAP2110 - Projeto 4 - Tomografia Computadorizada

import numpy as np
from PIL import Image


# funcao que recebe a matriz dos coeficientes a_ij
# o vetor dos valores b_i
# o numero maximo de ciclos do algoritmo p_max
# e o vetor inicial x_k
# e retorna o vetor final com as densidades x_j
def vetor_densidades(a, b, p_max=100, x_k=None):

    # m é a quantidade de b_i's
    # n é a quantidade de a_j's
    m = a.shape[0]
    n = a.shape[1]

    # se nenhum ponto inicial é especificado
    # inicia com a origem
    if x_k is None:
        x_k = np.zeros((n,1))
    
    # loop até p_max ciclos
    for p in range(p_max):

        # iteracao nos m hiperplanos
        for k in range(m):

            # projecao do ponto no hiperplano
            a_kT =  a[k]
            a_k = a_kT[:, np.newaxis]

            numerador = b[k:k+1] - np.matmul(a_kT, x_k)
            denominador = np.matmul(a_kT, a_k)
            x_k += (numerador / denominador) * a_k

            x_kT = x_k.transpose(1, 0)

            # print para visualizacao de cada iteracao
            #print((p, k), x_kT)

    return x_kT


# funcao que cria uma imagem a partir de um vetor de densidades
# remodelar permite transformar o vetor 1D em um array 2D
# pode redimensionar para aumentar a imagem
# e pode criar uma imagem proporcional a razao dos fotons em vez da densidade
def imagem_tomografia(densidade, remodelar=None, redimensionar=None, proporcional_fracao=False):

    if not proporcional_fracao:

        x_max = np.amax(densidade)
        densidade = 1 - (densidade / x_max)

        # transforma o vetor de densidade com valores entre 0 e 1
        # em um vetor de graus de cinza com valores entre 0 e 255
        # sendo 0 mais escuro e 1 mais claro
        vetor_imagem = densidade * 255

    # transforma o vetor de densidades em um
    # vetor de fracao de fotons que passam sem serem absorvidas
    else:
        vetor_imagem = np.exp( - densidade)
        vetor_imagem = vetor_imagem * 255

    # remodela o vetor
    if remodelar is not None:
        vetor_imagem = np.reshape(vetor_imagem, remodelar)

    # print para visualizacao do vetor
    #print(vetor_imagem)

    # transforma o vetor em imagem
    imagem = Image.fromarray(vetor_imagem)

    # redimensiona a imagem
    if redimensionar is not None:
        imagem = imagem.resize(redimensionar)
    
    # print para visualizacao da imagem
    imagem.show()

    return imagem


# exemplo das 3 retas
'''
a = np.array([[1, 1],
            [1, -2],
            [3, -1]])
b = np.array([2, -2, 3])
d = vetor_densidades(a, b, p_max = 6, x_k = np.array([[1.], [3.]]))
imagem_tomografia(d, redimensionar=(500, 250))
'''

# imagem da questao 3 (centro do pixel)

a = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 1, 0]])

b = np.array([8.00, 15.00, 13.00, 14.79, 14.31, 3.81, 18.00, 12.00, 6.00, 10.51, 16.13, 7.04])

d = vetor_densidades(a, b, p_max=45)

imagem_tomografia(d, remodelar=(3, 3), redimensionar=(300, 300), proporcional_fracao=True)