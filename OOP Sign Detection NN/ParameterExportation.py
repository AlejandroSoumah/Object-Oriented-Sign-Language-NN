# NumberOfWeights:4
# NUmberofBiases:4
# WeightPath:'C:\\Users\\aleja\\Desktop\\DeepLearning\\Parameters\\SignDetection\\Pruebas\\
# BiasesPath:'C:\\Users\\aleja\\Desktop\\DeepLearning\\Parameters\\SignDetection\\Pruebas\\
# NameofWeights :parameters_w
# NameofBiases :parameters_b
def ParameterExportation(NumberOfWeights , NumberofBiases , PathWeights , PathBiases , WeightName , BiasesName ):
        for i in range( 1 , NumberOfWeights ):
            np.savetxt( PathWeights + WeightName + str(i) + '.dat' , parameters["W"+str(i)] , delimiter=',')
        for i in range( 1 , NumberofBiases ):
            np.savetxt( PathBiases + BiasesName + str(i) +'.dat', parameters["b"+str(i)] , delimiter=',')
        print(" Weights and Biases have been exported correctly !!")
#ParameterExportation(4 , 4 , 'C:\\Users\\aleja\\Desktop\\DeepLearning\\Parameters\\SignDetection\\Pruebas\\' , 'C:\\Users\\aleja\\Desktop\\DeepLearning\\Parameters\\SignDetection\\Pruebas\\' , 'parameters_W' , 'parameters_b' )
