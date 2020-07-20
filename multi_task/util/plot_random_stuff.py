from matplotlib import pyplot as plt

if False:
    plt.figure(figsize=(9, 4))
    plt.title('Percentage of active connections per layer')
    plt.plot([24, 23, 28, 12, 9, 9, 9, 5, 5, 4, 3, 1, 1, 1, 13], label='00_22_on_June_10 (bias;comeback)')
    plt.plot([21, 21, 25, 11, 9, 9, 9, 5, 4, 3, 3, 1, 1, 1, 12], label='20_33_on_June_10 (bias;nocomeback)')
    plt.plot([20, 20, 24, 11, 9, 8, 8, 4, 4, 3, 2, 1, 1, 1, 10], label='21_26_on_June_12 (freezebias;nocomeback)')
    plt.plot([18, 19, 23, 10, 8, 8, 8, 4, 3, 3, 2, 1, 1, 1, 10], label='17_31_on_June_14 (nobias;nocomeback)')
    plt.plot([1.978, 1.172, 1.123, 0.464, 0.415, 0.458, 0.406, 0.168, 0.136, 0.107, 0.085, 0.020, 0.014, 0.013, 2.031],
             label='17_35_on_May_20 [pretty sparse]')
    plt.plot([0.903, 0.513, 0.635, 0.189, 0.189, 0.134, 0.104, 0.085, 0.079, 0.067, 0.043, 0.016, 0.014, 0.014, 5.483],
             label='14_53_on_May_26 [sparsest]')
    plt.plot([41, 43, 56, 36, 31, 33, 36, 21, 14, 13, 17, 6, 6, 4, 34],label='02_16_on_May_23 [not sparse]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.title('Number of active connections per layer')
    plt.gca().set_yscale('log')
    plt.plot([972, 928, 2282, 1966, 1534, 1502, 3007, 3488, 3097, 2660, 4489, 3449, 2728, 1898, 2569], label='00_22_on_June_10 (bias;comeback)')
    plt.plot([855, 846, 2042, 1779, 1483, 1425, 2980, 3185, 2797, 2285, 3851, 2793, 2263, 1692, 2488], label='20_33_on_June_10 (bias;nocomeback)')
    plt.plot([833, 806, 1933, 1796, 1406, 1389, 2631, 2925, 2565, 2164, 3164, 2523, 1990, 1519, 2038], label='21_26_on_June_12 (freezebias;nocomeback)')
    plt.plot([754, 762, 1883, 1657, 1281, 1330, 2560, 2641, 2116, 1926, 2996, 2269, 1731, 1481, 2059], label='17_31_on_June_14 (nobias;nocomeback)')
    plt.plot([81, 48, 92, 76, 68, 75, 133, 110, 89, 70, 112, 53, 36, 34, 416],
             label='17_35_on_May_20 [pretty sparse]')
    plt.plot([37, 21, 52, 31, 31, 22, 34, 56, 52, 44, 57, 42, 38, 37, 1123],
             label='14_53_on_May_26 [sparsest]')
    plt.plot([1686, 1780, 4559, 5920, 5084, 5401, 11691, 13700, 9209, 8610, 21886, 15488, 15019, 10793, 6916],label='02_16_on_May_23 [not sparse]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
else:
    plt.plot([0.8319, 2.2558, 1.1102, 1.0518, 1.8588, 0.8392, 0.3220])
    plt.show()
    plt.plot([  4.1902,   2.1484,   0.7305, 494.3111,   6.5799,   8.0195,   0.8689])
    plt.gca().set_yscale('log')
    plt.show()
    plt.gca().set_yscale('log')
    #WRONG!!!!!!!!!!!!!!!
    plt.plot([ 0.8162,  0.3150,  0.6414, 25.9640, 14.5297,  0.8512,  0.2740])
    plt.show()
    plt.gca().set_yscale('log')
    plt.plot([0.2610, 0.1330, 0.2241, 3.2203, 5.3210, 0.2643, 0.0301])
    plt.show()
    plt.gca().set_yscale('log')
    plt.plot([0.1259, 0.0646, 0.1091, 1.6097, 2.6591, 0.1255, 0.0160])
    plt.show()
