from . import gcn_network as gcn
#import network.chebconv_network
#import network.splineconv_network

networks = {
                #'GCN_test' : gcn.GCN_test,
                #'GCN_32' : gcn.GCN_32,
                #'GCN_32_64' : gcn.GCN_32_64,
                #'GCN_32_64_128' : gcn.GCN_32_64_128,
                #'GCN_8': gcn.GCN_8,
                #'GCN_8_8': gcn.GCN_8_8,
                #'GCN_8_8_8': gcn.GCN_8_8_8,
                #'GCN_8_8_8_8': gcn.GCN_8_8_8_8,
                #'GCN_8_8_8_8_8': gcn.GCN_8_8_8_8_8,
                #'GCN_8_8_16': gcn.GCN_8_8_16,
                #'GCN_8_8_16_16': gcn.GCN_8_8_16_16,
                'GCN_8_8_16_16_32': gcn.GCN_8_8_16_16_32,
                #'GCN_8d_8d_16d_16d_32d': gcn.GCN_8_8_16_16_32,
                #'GCN_8_8_16_16_32_32' : gcn.GCN_8_8_16_16_32_32,
                #'GCN_8_8_16_16_32_32_48' : gcn.GCN_8_8_16_16_32_32_48,
                #'GCN_8_8_16_16_32_32_48_48' : gcn.GCN_8_8_16_16_32_32_48_48,
                #'GCN_8_8_16_16_32_32_48_48_64' : gcn.GCN_8_8_16_16_32_32_48_48_64,
                #'GCN_8_8_16_16_32_32_48_48_64_64' : gcn.GCN_8_8_16_16_32_32_48_48_64_64,
                #'GCN_8bn_8bn_16bn_16bn_32bn' : gcn.GCN_8bn_8bn_16bn_16bn_32bn,
                #'GCN_4_4_8_8_16_16_32' : gcn.GCN_4_4_8_8_16_16_32,

                #'ChebConv_test' : network.chebconv_network.ChebConv_test,
                #'ChebConv_8_16_32' : network.chebconv_network.ChebConv_8_16_32,

                #'SplineConv_test' : network.splineconv_network.SplineConv_test,
}

def get_network(name, numFeatures, numClasses):  
  return networks[name](numFeatures, numClasses)
