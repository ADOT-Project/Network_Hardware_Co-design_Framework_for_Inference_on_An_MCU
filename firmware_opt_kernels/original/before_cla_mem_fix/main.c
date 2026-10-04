

/**
 * main.c
 */
#include "F28x_Project.h"     // Includes everything: PieCtrlRegs, Cla1Regs, etc.
#include "cla_dense_layer_shared.h"
// GUNCELLEME (asama bazli bolusum): hangi asamanin CLA'da hangisinin CPU'da
// calisacagini belirler; weights.h'teki DATA_SECTION pragma'larini da
// suruyor, bu yuzden ONCE include edilmeli.
#include "cla_pipeline_config.h"
#include "weights.h"
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "convolution.h"
#include "definitions.h"
#include "feature_tables.h"
// GUNCELLEME: gercek ADC/DMA yolu henuz hazir degil (ayri bir gorusme
// konusu -- arabellek boyutu eski SLICE_LEN'e gore, cift tamponlama yok).
// Bu asamada ucu sinifin (Healthy/Bearing/Propeller) golden vector'u AYNI
// ANDA dahil edilip runtime'da aralarinda gecis yapilarak pipeline'in
// (mfcc_extract -> quantize -> CLA conv1+conv2+FC) girdinin degismesini
// dogru yakalayip yakalamadigi test ediliyor -- gercek ADC olmadan.
// GUNCELLEME (MIMII): golden vector seti modelin sinif sayisina gore
// seciliyor -- 2 sinif = MIMII fan (normal/abnormal), 3 sinif = BLDC.
// ADOT her export'ta NUM_CLASSES'i definitions.h'e yaziyor, yani hangi
// veri setiyle egitilen model port edilirse edilsin dogru set derlenir.
#if NUM_CLASSES == 2
#include "golden_vector_normal.h"
#include "golden_vector_abnormal.h"
#elif NUM_CLASSES == 3
#include "golden_vector_healthy.h"
#include "golden_vector_bearing.h"
#include "golden_vector_propeller.h"
#else
#error "Bu sinif sayisi icin golden vector seti yok (2 = MIMII, 3 = BLDC)"
#endif
#include "adc.h"

#define INPUT_SIZE 2704

#ifdef __cplusplus
#pragma DATA_SECTION("CpuToCla1MsgRAM");
float fVal; //input
#pragma DATA_SECTION("Cla1ToCpuMsgRAM");
float fResult;  //output
#else
#pragma DATA_SECTION(cla_count_param,"CpuToCla1MsgRAM");
uint16_t cla_count_param; //input

#pragma DATA_SECTION(cla_offset_param,"CpuToCla1MsgRAM");
uint16_t cla_offset_param; //input

#pragma DATA_SECTION(fResult,"CLADataLS0");
float fResult[NUM_CLASSES];  //output

#pragma DATA_SECTION(input_vector,"CLADataLS0");
int16_t input_vector[DENSE_LAYER_INPUT];

//#define MAX_ELEMENTS 11492 // 16 * 26 * 26
// GUNCELLEME (asama bazli bolusum): workspace (pool1 + pool2 ciktilari) HER
// konfigurasyonda CLA veri RAM'inde tutulur -- hicbir asama CLA'da olmasa
// bile. Sebep olculdu (TI cl2000 22.6.1 ile 8 kombinasyonun hepsi derlendi):
//   * CPU RAM'i (RAMGS_TOTAL_CNN, 45056 word) MFCC tamponlariyla zaten
//     42688 word dolu (mfcc_extract::x[] tek basina 33792 word). workspace
//     (2532 word) oraya dusurulunce ".ebss will not fit" (#10099) ile link
//     hatasi alinir (cla_pipeline_config.h: 0/0/0).
//   * CLA veri RAM'i (RAMLS_0_1_2_3_4, 10240 word) hicbir kombinasyonda
//     sikisik degil (en fazla 6858 word, hepsi CLA'dayken).
// CPU, CLA veri RAM'ini (LSx, MSEL=CLA) zaten okuyup yazabiliyor; kisit tek
// yonlu -- CLA bu havuzun DISINA erisemiyor. Yani tamponu burada tutmak her
// kombinasyonda dogru, CPU'ya tasimanin kazanci yok, bedeli link hatasi.
// (Asamaya gore kosullu olan sadece AGIRLIKLAR: CPU'dayken flash'ta kalirlar
// ve RAM harcamazlar.)
#pragma DATA_SECTION(workspace, "CLADataLS0");
#pragma DATA_ALIGN(workspace, 8)
//float workspace[MAX_ELEMENTS] = {0.0f};
 int16_t workspace[MAX_ELEMENTS] = {0.0f};

// GUNCELLEME: 864 elemanli eski (32x12 girisli model) veri, 2016 elemanli
// yeni (65x12 girisli, dataset__eval_heldout_test) fc_weights ile degistirildi.
// GUNCELLEME 2 (GERI ALINDI): "CLADataLS1" pragma'si tekrar eklendi -- FC
// hesabi yeniden CLA'da (bkz. dense_layer.cla). Asil sorun MAX_ELEMENTS'in
// (workspace) eski, fused-olmayan tasarimdan kalma yanlis formulle ~9000
// byte israf etmesiydi (definitions.h::MAX_ELEMENTS notu); o duzeltilince
// weight[] (4032 byte) CLA'nin dar belleğine rahatlikla sigiyor.
// GUNCELLEME (asama bazli bolusum): FC agirliklari (4032 byte) yalnizca
// dense asamasi CLA'dayken CLA RAM'inde tutulur; CPU'ya alinirsa flash'ta
// kalir ve CLA havuzunda 4032 byte yer acilir.
#if CLA_STAGE_DENSE
#pragma DATA_SECTION(weight,"CLADataLS1");
#endif
const int16_t weight [DENSE_LAYER_WEIGHTS] = {
    4873, -7570, -12130, 19025, 3380, 5624, -565, 10528, 756, -3205,
    -5931, 18289, 662, 9207, 2795, 5906, -1620, -12335, -5737, 14327,
    -7547, 17707, 3937, 5537, -404, -1395, -9437, 21993, -3860, 6816,
    14539, 12222, -3188, 862, -11867, 12809, 3009, 5959, 5934, 8825,
    -5352, -12473, -13467, 12928, -6982, 13571, 1549, 5332, -783, -1637,
    825, 7117, -9148, 9108, 4942, 12284, 1523, -7632, -4438, 13162,
    -4193, 7188, 8122, 3648, 2202, -13136, -4912, 13486, -3398, 9654,
    6954, 5058, -724, -5096, -5659, 13993, -8083, 6060, 3291, 13035,
    6740, -10053, -2772, 6691, 1631, 14186, 786, 4322, 2044, -9339,
    -709, 20876, -4932, 9358, 12448, -2080, 8015, -1554, -11677, 22472,
    -4659, 10148, 6524, 5668, 2525, -12923, -13008, 20896, -6567, 14133,
    3770, 8589, 7368, 2231, 6817, 2230, 3264, -4875, 5848, 12834,
    -1679, -9580, 9296, -339, 2571, 1096, 3500, 4448, -1312, -8734,
    5054, 6793, -10407, -6174, 1117, 6802, -305, -317, 2849, 70,
    -672, -3486, 2497, 6337, 7206, -419, 7074, 4245, -2617, -3875,
    4806, 12883, 11375, -8607, 3664, 751, -10061, -18113, 2364, 1161,
    2433, -10228, 8891, 5458, -14322, -4880, 9765, -3078, 7571, -2569,
    12622, 11313, -5382, -2902, -1648, 4341, 3547, 7948, 8386, 7385,
    -5651, 1641, -82, 5294, 1759, 7251, 1800, 1929, 5293, -1057,
    868, 3223, -1176, -9935, 457, 5878, -1809, -2539, 3624, 9087,
    -3747, -3193, 9023, -454, -9794, -12887, 2838, 10567, 7248, 6198,
    10822, 2223, -392, -8407, 6359, 15280, 1409, -5574, 14885, 3414,
    -509, -5204, 7858, 6632, 7463, 1589, -22476, 2276, -12628, 1923,
    -10631, -5366, -3966, -1277, -17910, -3054, -8792, 2029, -17457, -632,
    1099, -2660, -21764, 2008, -1377, 9326, -21417, -5508, -8700, -2142,
    -16822, 1182, 3042, -4, -23748, -2550, -2562, 3977, -11602, 4288,
    -7172, 12186, -16380, -5712, -114, -3624, -9808, 10101, -14778, 11644,
    -28821, -9360, 2001, 647, -9842, 1766, -12660, 5953, -27756, -10138,
    7563, 9423, -15471, -3288, 4036, 3108, -18895, 6010, 3450, 6865,
    -22323, 7053, -3348, 5143, -1986, -7906, -104, -8809, -23549, -6905,
    -8918, 1298, -14570, -9393, -2073, -1727, -15416, -1964, -9057, 6194,
    -17007, -5649, -4510, -2323, -15985, -3228, 943, 10199, -26641, -1173,
    -6602, 2606, -9163, 12011, -8565, 17890, -16804, -9492, -1895, 11903,
    -7131, 11775, -11280, 13185, -29960, -7389, 2210, -10219, 1051, -11090,
    23978, -107, 2601, -8373, 960, 668, -5526, -10024, 4215, -11739,
    11710, -4796, -2652, -4494, -10963, -4606, 18184, -15096, 9341, -1748,
    -2514, 4425, -14733, -6038, 1642, -2234, 7259, 5893, -2698, -639,
    -29885, -3941, 16584, -6229, 8943, -3864, 298, 391, -7875, -4975,
    29808, 446, 6874, -2571, -7260, -609, -5528, -11809, 13050, 3143,
    -7634, 6445, -754, 694, -3612, -3727, 15875, -1225, 4692, 2923,
    -3383, -6680, -9993, -4458, 20133, 1279, 3684, 2342, 2455, -1797,
    -1835, -4219, 11554, -2651, 7294, -6528, -4654, -9434, -7111, -8559,
    20428, 4717, 9290, 6074, 634, 7236, -7584, -11442, 12885, -9747,
    5850, 3791, -4259, 3293, -21899, -4438, 13984, -4633, 6438, -333,
    4121, 4672, -12406, -110, 27700, -4813, 7373, 1600, 5071, 6618,
    -13147, -9882, 18249, 5221, 20839, 18151, -3553, 1135, -20740, -20718,
    6059, -279, 24052, -9848, -2393, 3691, -23417, -7533, 25980, -2889,
    2148, -9190, 5907, 6516, -13769, -4628, 15676, 9170, -4070, -1825,
    9492, 2106, -23925, -10169, 769, -3779, 1625, 2881, 14237, 2233,
    -17591, 5759, 15691, 10591, 14930, 10320, 11931, 16129, -30740, 1916,
    11412, -8896, 10442, -5420, 6832, -2512, -28638, 5580, 7815, 1861,
    23076, 13066, -2097, 5423, -16203, -2905, 10023, -6245, 12333, 3235,
    7927, 1743, -17929, -4876, 5903, -6686, 16814, -30, 2520, 4018,
    -10561, -1212, 3249, -7549, 11968, -7563, 2474, 3752, -7196, -4452,
    16099, 1753, 19042, -83, 6745, 8196, -18333, 2424, -1806, -7124,
    9422, 3452, 21598, 12326, -24198, -9897, 16985, -428, 12680, 2886,
    3086, 1379, -15, 5362, -17012, 2706, 7216, 15676, -4348, -8452,
    -15644, -2037, -1026, -944, -7735, 5595, -2794, -2717, 2578, -8872,
    -7334, 4079, 202, 12442, 227, -1119, 3373, -492, -5800, -4856,
    5945, 12849, -8286, -6765, 5592, 6654, -6620, -2226, 6697, 19680,
    -1510, -2881, -5399, 6930, -7271, 11181, 951, 2155, -11176, -10827,
    -4223, -119, -15790, -432, 7471, 7927, -6354, -12831, 644, 7266,
    -1097, 6324, -2179, 10866, 3320, -7262, 63, 10473, -2660, 453,
    2485, 14206, 91, -3423, 453, -872, -4982, 4084, 1584, 7448,
    695, -11235, 2526, -7156, -4626, -6829, 3774, 16686, 1717, -1743,
    1642, -10191, 5036, 5924, 17431, 13364, 759, -7434, 9844, 10709,
    -4074, -2269, 2768, 16645, -3806, -1463, 8011, 4636, -16485, 17440,
    1512, 10515, -14150, 2967, 7938, 19389, 21504, 688, -1490, 23143,
    -12660, -8592, 14941, 21604, 5942, 3206, -2539, 19151, -6523, 4340,
    13378, 6650, -1516, -2973, 13930, 21911, -5244, 16772, 7556, -545,
    9785, -977, -2400, 8249, -7548, -3109, -21172, 1444, -2662, -15948,
    7063, 14314, -5390, 2347, -1024, 1478, 4561, -1393, -3879, 18848,
    -18240, -1386, 6407, 2443, -2271, -2320, 13226, 18094, -10053, -2010,
    16239, 11923, 3993, 2971, 18929, 21393, -7096, -993, 10616, 15381,
    14852, 6725, 9541, 17281, 2229, 3613, 8017, 10346, -841, -1308,
    8826, 18964, -11649, -6642, 6725, 10289, -3160, 5051, 3850, 19099,
    -5295, 17421, 8744, -7671, 1336, -3046, -10720, 12079, -11938, 15454,
    -9409, -387, -7564, -8557, 8773, 12702, -4136, 1985, 7567, 5109,
    8417, 4236, 12634, 3062, -6649, -14616, 8692, 1730, 3465, -869,
    6661, -6107, 1101, -21870, 8816, 1139, 3672, 1630, -9115, -3237,
    -9934, -14461, 6598, -3787, -11532, 352, -4303, -5477, -7438, -21518,
    -5653, 832, -2275, 1761, 4646, -6786, -1583, -20455, 10782, -9837,
    2486, -1487, 3544, 5163, 1956, -19380, -2838, -1000, -2359, 2702,
    6919, -1733, -7769, -21632, -1799, -9289, 5093, 729, -652, -4536,
    -2578, -18699, 12043, -2031, -365, -7757, 9042, 2565, -9684, -2947,
    12490, -2187, -141, -6587, 56, -17143, 505, -7408, 1375, -4241,
    -817, 2717, -776, -7458, -6274, -22807, 1737, 1160, 834, -3938,
    -6751, -8684, -4797, -24128, -9473, -16465, -1049, 3267, -1633, -4215,
    -3211, -12807, 6962, -6159, 1005, -5664, 3908, -2112, -4452, -20524,
    3547, -5229, -1453, 7188, 6747, -473, 5951, 7681, 2686, -13944,
    -4706, 792, -7035, -8628, 1288, 6899, 2374, -25356, -28004, -10631,
    -19482, -40, 1387, 371, 2687, 5121, -5245, 2139, -10657, -8580,
    1123, -9028, 12105, 7282, 1908, -7604, -13139, -7289, 8381, 1415,
    8786, -18155, -7006, -17733, -2285, 7750, 20409, 7667, 1001, -16586,
    -6744, 2285, -7753, -11684, 30595, -11442, 9401, -9201, -9595, -3236,
    -9530, -17346, 17666, -7338, -5829, -13119, -3166, -16815, -11229, -1887,
    -12754, 5165, 2940, -9862, -21291, 9879, -15696, -10115, 1147, 12230,
    -4680, -7173, -15134, -12192, -13177, -12500, -5046, 6808, -3299, -3150,
    -15007, 338, -18503, -7469, -4719, 10347, 5284, -14970, 1026, -4504,
    -11696, -3933, 12834, -517, 5822, -11617, -5882, -7897, -13452, -459,
    12112, -2855, 19087, -16415, -9231, 6760, -4119, -2372, -32678, -1099,
    10380, 9075, -6364, 5092, 3248, -2506, -17712, -1516, 12792, 4699,
    -6815, -5109, -10984, -4745, -11917, -4154, 12198, 5211, 789, 1400,
    -7184, 2199, -3765, 5548, -3922, 17799, 603, 12349, 2595, -1040,
    -26527, -3079, 10520, 11966, 1159, 3916, -1783, 4446, -8913, 3672,
    9865, -2546, -953, -521, -2491, -6075, -13025, -1094, 3709, 7991,
    -222, 2463, -1502, -4983, -24052, 3868, 10608, 5480, -4498, -7160,
    -1024, -215, -11928, -6020, 10486, 8152, -200, 10218, 3243, -1492,
    -14408, -9702, 966, 5491, -6283, -1969, -4873, 8483, -17370, -6137,
    3084, 6473, -4753, 3129, -2762, 176, -6830, -6136, -2164, 10823,
    4216, 5826, 1044, 10488, -14948, 1560, 7350, 8837, 1829, 4424,
    -5546, 3174, -25397, -661, 8189, -1096, -5645, 4558, 290, -7179,
    -14605, -2519, 1956, 12186, -7199, 855, 7265, 2300, -12399, -6565,
    -4099, -2929, -5833, 17755, 4482, 6037, -14848, -7605, 1925, -260,
    3392, 8206, -1629, -3047, -2663, -5417, 7248, 792, -3, 5849,
    10039, -1084, -6294, -10878, -8044, 3777, -725, 4740, 9513, 6640,
    -7172, -6241, -1782, 6449, 1533, 2387, 11834, -2622, -6990, -12080,
    2945, 1989, -8678, 4164, 4746, -3077, -16398, -11935, 4677, 2117,
    -3786, 4158, 340, 568, -7333, -5598, -7323, 5522, -4548, 2820,
    7204, -1100, -12221, -11607, 1175, 2575, -5607, 1920, 6158, 7208,
    -12236, -8972, -5570, -3266, -3854, 3261, 931, -4088, -3435, -4598,
    -3076, 1425, -4160, 4967, -321, 4779, -4822, -2218, 1646, 1464,
    -5680, -756, 10228, -195, -12691, -16459, 9237, 8110, -6066, 2647,
    12256, 6650, -7460, -13697, 18448, -14389, 25, 1100, 1064, -500,
    88, -19541, 2491, -8139, 3063, 2792, -7242, 6925, 245, -20533,
    895, 1566, 4370, 1204, -931, 10255, -3065, -8907, 2099, -9335,
    7541, 4703, 122, 3166, -2977, -12554, -5912, 7022, 5137, -3745,
    4175, 8184, -1502, -7272, 1351, 80, 4648, 1748, 11247, 8595,
    4873, -9353, -2545, 1314, 3962, 366, -2562, -4207, -3014, -19496,
    13980, 10655, 2449, 1406, -7235, -6252, -3064, -12101, 11727, -511,
    9876, 2422, -5261, 3191, 2331, -11405, 6020, -1570, 4834, 10482,
    -8950, 6078, -5714, -22815, 1692, 2272, 4252, 4973, 113, 12990,
    251, -13841, 7112, 651, 1577, 146, -490, 13248, -5354, -5739,
    1626, -721, 5110, -6258, -829, 6409, -3635, -8335, 9617, -991,
    8634, 6266, 3806, 1153, 164, 5116, 8579, -18547, -4294, -5085,
    3090, -6207, -1254, 1164, 6809, -14576, -2717, -5261, -5071, -5257,
    -1946, 13357, 1650, -13577, 4567, -18475, -8063, -3297, -1865, 3564,
    11374, -23819, 3811, -8901, -12248, -10193, 5565, 148, 8582, -9833,
    -3502, -5805, -9409, -9942, 5013, 13651, 13302, -12873, 7530, -13809,
    -2326, -5404, 874, 223, 161, -10538, 4480, -10558, -6419, -6174,
    -4397, 6577, -1892, -13620, 5274, -3758, -5328, -8554, -647, 12214,
    7339, -13347, -1449, -9857, -5707, -2426, 3598, 4457, 7806, -13974,
    6948, -3679, -5791, -11339, -3665, 12177, 2118, -3012, 301, -8001,
    -763, -4037, -7414, 3108, 3273, -22041, 6344, -11672, -9946, -1597,
    -6168, -315, 15485, -20938, 488, -11200, -7929, -6571, 79, 12900,
    13154, -22644, 8266, -14548, -2046, -3912, -5836, -905, -3993, -1834,
    -844, 10100, -6977, -11580, 1102, 4454, -3661, 3761, 1475, -2711,
    1554, -5229, 5063, 11795, -4934, -12006, 8927, 12135, -1624, -12894,
    -1796, -281, -5157, -1447, 1523, 3177, -869, -4401, -5379, 1287,
    -13820, -4824, 3497, 3901, -10439, -12363, -11401, 5232, -7215, 88,
    11278, 15569, -7977, -3959, 2099, 14229, -6320, -7035, 13501, 2423,
    -5632, 4292, -6574, 6131, -11242, -11418, 6790, 2552, -938, -2304,
    -1993, -8143, -9692, -4245, 1264, 1815, -4815, -6521, -2398, -1384,
    -4805, -3873, -7195, 1669, 589, 880, 335, 6946, -4567, -7373,
    3088, -827, -3657, -5548, -763, -426, -10937, -3632, 14498, 7666,
    -999, -8629, -2571, -5271, -11463, -848, 3313, 2282, -7404, -11419,
    -1971, 8776, -13199, -3337, 2730, 6538, -14670, -8260, -7076, -4914,
    22941, -19, 13781, -3613, 13406, 5548, 8810, 3544, 20891, 1813,
    9256, 2311, 22359, 7184, -2601, 361, 22775, 2538, 5645, -13261,
    19075, 5672, 5687, 179, 20165, -1201, -3883, -5999, 20367, 1302,
    6038, 171, 11288, -9383, 6728, -11666, 16400, 7232, -4944, 1865,
    8160, -10262, 14436, -9288, 30517, 9532, -2066, -3888, 11995, 2112,
    13650, -9121, 24811, 15440, -3439, -7160, 12916, 3856, -8666, -6230,
    18915, -7736, -5979, -6551, 27505, -4254, 2573, -7189, -174, 9011,
    2255, 6145, 18772, 8439, 8123, 698, 14158, 8118, 16, 2456,
    17663, 1085, 7045, -7131, 15763, 4171, 2699, -1584, 21837, 101,
    27, -10594, 29111, -4096, 6147, -5004, 2987, -13241, 6112, -15092,
    18214, 12454, -165, -8422, 11551, -5419, 10094, -19626, 29441, 10356,
    -1333, 6791, 3539, 14481, -21101, -2558, -1777, 7613, -2618, -6209,
    2919, 13066, -6469, 8018, -8230, 9775, 2696, 2929, 9994, 6650,
    -11289, 10352, -6747, -4176, 4734, -4482, 14876, 6095, -3689, 6286,
    -4337, -1525, 1512, 2527, 28726, 9649, -16372, 8220, -12106, 1472,
    -767, -4017, 5230, 7502, -30306, 5495, -7136, 2328, 6056, 6092,
    6643, 10211, -15352, 885, 6468, -4784, 80, -3157, 4205, 8468,
    -19238, -1269, -1773, -1177, -341, 7414, 9348, 7618, -21846, -2479,
    943, -4860, -1512, 4516, 541, 3865, -12097, -159, -8770, 6178,
    5512, 9561, 9385, 5318, -16921, -3068, -5339, -9284, -393, -5210,
    11559, 14308, -9961, 8042, -9036, -8961, 1674, -6077, 23421, 6379,
    -13040, 5084, -8364, -3071, -4818, 1117, 13398, -2934, -28634, 8774,
    -5542, 385, -9217, -10650, 15374, 11387, -14264, -4971, -23602, -17501,
    4011, 4268, 21448, 19153, -4766, 4212, -27960, 10781, -2927, -4732,
    21246, 5208, -24841, 361, -4010, 7638, -10174, -3465, 12814, 6585,
    -15265, -6302, 4067, 6500, -11830, -2014, 19433, 8702, -775, 8279,
    -2422, -148, -10365, -5627, 13499, -2183, -14141, -13777, -19441, -8568,
    -10011, -13277, 30189, 1348, -6985, 7348, -16671, 6943, -2758, -957,
    27815, -9640, -2978, -2356, -29934, -13039, -1547, -3359, 21771, 4607,
    -8053, 6029, -11837, -3594, -2375, -444, 17147, 3170, -7227, 6707,
    -20247, -2220, 922, -3664, 16381, -2509, -2194, 9349, -15121, 11023,
    -3268, -6601, 6848, 7663, -10151, -167, -12885, 613, -6002, -6167,
    15604, -1981, 7596, 6370, -6754, -495, -21834, -14150, 20159, 10524,
    -15300, 2070, -16629, -3486, -2809, -1497, 1256, -6168, 17112, -3310,
    -3232, -13857, 5935, 6689, 10835, 3599, -1011, 4661, 4856, -6087,
    1041, 4073, -1160, 13785, 3899, -3781, -2015, -15002, -1378, 1588,
    -493, 1849, 2814, 1555, -6563, -13600, 11118, 5634, -5751, -4360,
    7760, 774, -4748, -17573, 144, 3555, 4112, -6872, 5778, -7951,
    96, -1958, 11441, 5683, 9209, -1434, 15280, 4059, -628, -6536,
    8103, 12953, -1430, -9301, -783, -4962, 1665, -9082, -4162, 8757,
    -2134, -10884, 8645, -842, -2331, -12108, 252, 2585, -2123, -4330,
    5498, -47, -770, -13510, -2075, 12893, 511, 1625, 4023, 808,
    -3903, -13796, -2545, 416, -3740, 4838, -5819, -1992, -11817, -9146,
    -1570, 11759, -7495, -8533, 5104, 2947, -6766, -20095, 5346, 1979,
    -4715, -5848, 13558, -17551, -2243, -9896, 10563, -1347, -6522, -17125,
    -18648, -6971, -4256, -26684, 7353, 4366, -14112, -18101, -6284, 1157,
    1326, -18219, 11276, -4778, -15810, -5189, 2705, 7086, -10233, -16614,
    11423, -15359, -6465, 4576, -7289, -593, -1689, -5547, 3469, 7248,
    20499, -2641, 2416, 12632, -7859, -16608, 2874, 4480, 1296, -2855,
    -4681, 2560, 2102, -21846, 22100, 5003, 71, -62, -420, 1538,
    -13742, -13389, 12495, 1470, -18824, -17654, 1444, -2425, -21440, -20055,
    11653, 1828, -12886, -20092, -12082, -4903, -5141, -18942, 721, -1997,
    -11863, -13729, 806, -365, -7765, -21625, 10527, 1866, -6527, -13806,
    4595, -5218, -5267, -18859, 3236, -13427, -7451, 3249, -3912, 24,
    8461, -14044, 5990, -16712, 6602, -2360, 7539, 8503, -8753, -15826,
    1759, -6064, -8259, -9517, -12566, 379, -11749, -2535, 6628, 19650,
    -7048, -1982, -4920, 2720, -7092, 8579, -2066, 21169, -13400, 2996,
    -1656, 1956, 5177, 6263, 13695, 18175, -2654, 5270, 5504, 844,
    1317, 7911, 11798, 18454, 1274, -2203, 1564, 749, -4754, 7221,
    2602, 24155, -10189, 9371, -4172, 5302, -5774, -8082, 3541, 20498,
    4593, 1161, -3274, -3850, -535, -1631, 5456, 16357, 1237, 8101,
    -9357, -1895, -1664, 1819, -3439, 13776, -9771, -263, -2614, 1321,
    -6821, -1456, 5633, 5283, -12485, 7419, 4090, 4973, 1529, 16725,
    942, 3236, 1101, 4149, 622, -4441, 500, 6049, 7172, 20872,
    -2183, -5177, -2052, 1839, 4334, 4627, 4275, 21166, 10325, 14935,
    -6, -5494, -1649, 5246, 2322, 10029, -3081, 7282, -2399, 8605,
    -4360, 364, 5643, 16516, -1742, 7643, 1427, -13375, -3811, -4655,
    -5610, -8447, -8596, 7163, 10089, 1760, 7836, 8796, 3702, -4409,
    -428, 22723, 24760, 5495, 21945, -4688, -4679, 857, -2503, -4255,
    5672, -1627, 9951, 10030, -1224, 10795, -12403, -8467, -4506, 4936,
    14882, 4454, -6314, 297, -9955, 17490, 7918, 16242, 1663, -11383,
    -23362, -5247, -223, 15994, 10659, -5477, 9493, 12430, -32767, 13396,
    -12705, 14325, 6236, 2165, 9983, 16947, -16561, 4636, 8533, 8036,
    7900, 15492, 13737, 1758, 10042, -11971, 2035, 9223, 21222, -7790,
    17770, 12623, -2237, -11345, 8916, 9745, 11000, 10939, 15366, 12614,
    -1515, -3879, -1983, 6182, 14789, 1266, 12362, 3271, 2209, -9537,
    -5591, 12016, 302, 5977, 12485, 3222, -6565, 78, -11325, 5674,
    2394, 4838, 8120, 5020, -12254, -2770, -21631, 16645, 10945, -11160,
    8342, 3030, 31068, -1748, -11355, -6265, 5726, -3203, -1139, 4609,
    16373, 5619, -8653, -5023, 8052, 7699, 7816, 2085, 12290, 2817,
    -8841, -8568, 350, -3227, 771, -7376, 10230, -207, 2944, -18407,
    -385, -8518, 461, 1624, 21085, 3379, -6041, -11713, -4203, -5371,
    4705, -2478, 14809, 2317, -7938, -3066, 2167, -2072, 7743, 4325,
    17068, 4434, -2185, -7251, -642, -3918, 5843, 9852, 17632, -5952,
    -12266, -4452, 2282, 9101, 2923, 6439, 14597, 7256, -16494, -13831,
    -2256, -7844, -7609, -331, 14728, 6771, -4474, -7033, 7814, -597,
    9039, -9701, 14622, 8812, -8759, -7089, 6588, -3718, 3376, 4361,
    3219, 7379, -369, -9300, -7276, -1131, -1084, -10175, 18463, -715,
    -6103, -7268, -2515, -4160, 8250, -4378, 21816, 2135, -11679, 7076,
    6105, -4458, -333, 7082, 11699, 4716, 853, -11683, 5910, -5842,
    -6395, -4092, 13500, 7314, 3068, -135, 2420, -15345, -6754, -5984,
    10256, 5676, -4504, -1606, -1737, -1661, 2212, -1816, 7659, 2712,
    -2734, -984, -3415, -6451, -11809, -3358, 5536, 10357, 6079, -299,
    4664, -1675, -5679, -4895, 1762, 3073, -600, -5984, -3225, -2099,
    -10760, 3130, 7483, 7773, -863, -2733, 4738, -5718, -6067, 549,
    10124, 10292, -786, -2173, 8016, -6788, -5617, -6293, 10828, 3581,
    5582, -3378, 2145, -4672, -7260, -2208, 5840, 8909, -1032, -3489,
    4329, -3068, -3407, -7368, 13141, 10978, 3062, -1788, -2384, -5475,
    -4792, 3242, 2110, 9756, -2942, -2786, 4497, -7979, -5192, -10419,
    8277, 4525, -7849, 3566, 2531, -2017, -11888, 315, 12126, 17222,
    -12031, -2780, 5079, 174, -12154, -6245, 3966, 18390, -22698, 13555,
    -145, -1389, 1302, 2343, 4293, 22482, -1987, 10865, -4089, -7221,
    1511, -8182, 2641, 21738, -5612, 2708, -375, 2996, -1272, -11514,
    266, 8531, -4772, 13021, -7621, 18, -134, -5948, 4376, 16987,
    -364, -2242, -5229, 5488, -8321, -4593, -3721, 12289, 570, 926,
    -8093, -7297, -10708, -8894, -2957, 9424, 2229, -3455, -304, 2021,
    3407, -959, 4863, 18842, -15750, -8815, -1085, -588, 8567, 7388,
    2133, 13171, -9406, -1770, -6670, -2169, 1526, -3824, 3404, 15485,
    -8566, 2295, 323, -11745, 5809, -6464, 1974, 20572, 745, -1318,
    2359, -2824, 4507, -13474, -998, 11274, -6600, 541, -7049, -1702,
    228, -15178, 782, 10458, -4668, -85, -496, 7614, 1419, -4802,
    1653, 10597, -13436, 2020, -6643, -1319, -5691, -1925
};

#pragma DATA_SECTION(adcData0, "ramgs0");
//#pragma DATA_SECTION(adcData1, "ramgs0");
Uint16 adcData0[RESULTS_BUFFER_SIZE];
//Uint16 adcData1[RESULTS_BUFFER_SIZE];
volatile Uint16 done;
// EKLENDI: CLA1_1_INT (Cla1Task1 BITTIGINDE ates lenen kesme) bunu 1 yapar --
// main loop artik fResult[]'i CLA henuz hesaplamayi bitirmeden okumuyor.
// Eskiden boyle bir senkronizasyon yoktu.
volatile uint16_t cla_done = 0;
#endif //__cplusplus

void CLA_configClaMemory(void);
void CLA_initCpu1Cla1(void);

static float features[FEAT_N_FRAMES][FEAT_N_OUT];   /* 65x12 (eskiden 32x12 -- SLICE_LEN 2x oldu) */
__interrupt void cla1Isr1();

// EKLENDI: golden-vector dongusu -- gercek ADC yerine, uc sinifin
// (Healthy/Bearing/Propeller) onceden hesaplanmis golden_adc dizisi
// sirayla beslenip pipeline'in giris degisimini dogru yakalayip
// yakalamadigi (CLA sonucu -> beklenen sinifla eslesiyor mu) test edilir.
#if NUM_CLASSES == 2
static const uint16_t *const golden_inputs[NUM_CLASSES] = {
    golden_adc_normal, golden_adc_abnormal
};
static const char *const golden_names[NUM_CLASSES] = { "Normal", "Abnormal" };
#else
static const uint16_t *const golden_inputs[NUM_CLASSES] = {
    golden_adc_healthy, golden_adc_bearing, golden_adc_propeller
};
static const char *const golden_names[NUM_CLASSES] = { "Healthy", "Bearing", "Propeller" };
#endif
// CCS'in Expressions/Watch penceresinden debug sirasinda canli izlemek
// icin: hangi giris beslendi, model ne tahmin etti, dogru mu.
volatile uint16_t g_expected_class = 0;
// GERI EKLENDI: inference sonucu. run_inference() her pencereden sonra
// fResult[]'in argmax'ini alip bunlari gunceller:
//   g_predicted_class  modelin karari (0..NUM_CLASSES-1)
//   g_logits[]         o karar icin fResult[]'in kopyasi
//   g_correct_count / g_total_count   golden dongusundeki dogruluk
// CPU'da calisan (ya da karma) dagilimda sonuc run_inference() donunce
// hazirdir. Uc asama da CLA'daysa (CLA_SINGLE_TASK, paralellik modu) CLA
// beklenmez; o zaman bir ONCEKI pencerenin sonucu, CLA bitirmisse, bir
// sonraki tetikten hemen once kaydedilir -- yaris olmadan.
volatile uint16_t g_predicted_class = 0;
volatile float    g_logits[NUM_CLASSES];
volatile uint32_t g_correct_count = 0;
volatile uint32_t g_total_count = 0;
const char *volatile g_predicted_name = 0;

void run_inference(uint16_t expected_class);

void mfcc_extract(const unsigned int* adc_buf,
                        float out[FEAT_N_FRAMES][FEAT_N_OUT]);

int main(void)
{
    Uint16 resultsIndex;

    InitSysCtrl();
    InitGpio(); // Skipped for this example

    EALLOW;
    GpioCtrlRegs.GPCGMUX1.bit.GPIO67 = 0;
    GpioCtrlRegs.GPCDIR.bit.GPIO67 = 1;
    GpioDataRegs.GPCSET.bit.GPIO67 = 1;
    GpioCtrlRegs.GPCPUD.bit.GPIO67 = 0;
    GpioCtrlRegs.GPCCSEL1.bit.GPIO67 = 1;
    EDIS;

    GPIO_SetupPinOptions(18, GPIO_OUTPUT, GPIO_PUSHPULL);
    GPIO_SetupPinMux(18, GPIO_MUX_CPU1, 0);
    GPIO_SetupPinOptions(19, GPIO_OUTPUT, GPIO_PUSHPULL);
    GPIO_SetupPinMux(19, GPIO_MUX_CPU1, 0);

    DINT;
    // Initialize the PIE control registers to their default state.
    InitPieCtrl();

    // Disable CPU interrupts and clear all CPU interrupt flags:


    IER = 0x0000;
    IFR = 0x0000;

    InitPieVectTable();

    // ISR for ADCA INT1 - occurs after first conversion
    // ISR for DMA ch1 - occurs when DMA transfer is complete
    //
        EALLOW;
        PieVectTable.ADCA1_INT = &adca1_isr;
        PieVectTable.DMA_CH1_INT = &dmach1_isr;
        EDIS;

    //
    // Enable specific CPU interrupts: INT1 for ADCs and INT7 for DMA
    //
        IER |= M_INT1;
        IER |= M_INT7;

    //
    // Enable specific PIE interrupts
    //
    // ADCA INT1 - Group 1, interrupt 1
    // DMA interrupt - Group 7, interrupt 1
    //
        PieCtrlRegs.PIEIER1.bit.INTx1 = 1;
        PieCtrlRegs.PIEIER7.bit.INTx1 = 1;


        //
        // Stop the ePWM clock
        //
            EALLOW;
            CpuSysRegs.PCLKCR0.bit.TBCLKSYNC = 0;
            EDIS;

        //
        // Call the set up function for ePWM 2
        //
            ConfigureEPWM();

        //
        // Start the ePWM clock
        //
            EALLOW;
            CpuSysRegs.PCLKCR0.bit.TBCLKSYNC = 1;
            EDIS;

        //
        // Configure the ADC and power it up
        //
            ConfigureADC();

        //
        // Setup the ADC for continuous conversions on channels A3 and B3
        //
            SetupADCContinuous(&AdcaRegs, 3);
            SetupADCContinuous(&AdcbRegs, 3);

        //
        // Initialize the DMA
        //
            DMAInit();



    CLA_configClaMemory();
    CLA_initCpu1Cla1();

    EINT;  // Enable Global interrupt INTM
    ERTM;  // Enable Global realtime interrupt DBGM
    //
    // Initialize results buffer
    //
        for(resultsIndex = 0; resultsIndex < RESULTS_BUFFER_SIZE; resultsIndex++)
        {
            adcData0[resultsIndex] = 0;

        }

    //
    // Clearing all pending interrupt flags
    //
        EALLOW;

        DmaRegs.CH1.CONTROL.bit.PERINTCLR = 1;
        DmaRegs.CH2.CONTROL.bit.PERINTCLR = 1;
        AdcaRegs.ADCINTFLGCLR.all = 0x3;
        AdcbRegs.ADCINTFLGCLR.all = 0x3;
        EPwm2Regs.ETCNTINITCTL.bit.SOCAINITFRC = 1;
        EPwm2Regs.ETCLR.bit.SOCA = 1;

    //
    // Enable continuous operation by setting the last SOC to re-trigger the first
    //
        AdcaRegs.ADCINTSOCSEL1.bit.SOC0 = 2;
        AdcbRegs.ADCINTSOCSEL1.bit.SOC0 = 2;

        EDIS;

    //
    // Start DMA
    //
        done = 0;
        StartDMACH1();
        StartDMACH2();

    //
    // Finally, enable the SOCA trigger from ePWM. This will kick off
    // conversions at the next ePWM event.
    //
        EPwm2Regs.ETSEL.bit.SOCAEN = 1;

    // EKLENDI: sirayla beslenecek sinif indeksi (MIMII: 0=Normal,1=Abnormal;
    // BLDC: 0=Healthy,1=Bearing,2=Propeller; sonra basa sarar). static: fonksiyon disina tasinsa da her cagrida degerini korur.
    static uint16_t golden_cycle_idx = 0;

    for(;;)
     {
        GPIO_WritePin(18, 0);
        static uint32_t i,f,o,k;
        // GUNCELLEME: sabit golden_adc yerine, dongudeki mevcut sinifin
        // girisi besleniyor -- pipeline'in giris DEGISTIKCE dogru sinifa
        // gecis yapip yapmadigini (gercek ADC olmadan) test etmek icin.
        g_expected_class = golden_cycle_idx;
        mfcc_extract((const unsigned int*)golden_inputs[golden_cycle_idx], features);

        int idx = 0;
                for ( f = 0; f < FEAT_N_FRAMES; f++) {
                    for ( o = 0; o < FEAT_N_OUT; o++) {
                        float scaled = features[f][o] * INPUT_SCALE;
                        if (scaled > 32767.0f) scaled = 32767.0f;
                        if (scaled < -32768.0f) scaled = -32768.0f;
                        input_image[idx++] = (int16_t)scaled;
                    }
                }
        GPIO_WritePin(18, 1);

        // GPIO19: tamamen CPU'da (ya da karma dagilimda) dusuk kaldigi sure
        // INFERENCE SURESIDIR. Uc asama da CLA'dayken CLA beklenmedigi icin
        // yalnizca tetikleme ani kadar (mikrosaniyeler); o modda inference
        // suresi icin GPIO67'ye bakin -- onu Cla1Task1 kendi icinden suruyor.
        GPIO_WritePin(19, 0);

        // INFERENCE: run_inference() S1 -> S2 -> S3'u cla_pipeline_config.h'a
        // gore CPU'da ya da CLA'da calistirir ve sonucu (g_predicted_class,
        // g_logits, dogruluk sayaclari) kaydeder. Tamamen CPU'daki modelde
        // GPIO19'un dusuk kaldigi sure inference suresinin kendisidir.
        //
        // PARALELLIK MODU (yalnizca uc asama da CLA'dayken): CLA tetiklenir
        // ve BEKLENMEZ. CPU dongunun basina donup bir sonraki pencerenin
        // MFCC'sini hesaplamaya baslar; CLA bu sirada onceki pencerenin agini
        // isletir. Iki pin ayni anda dusuk gorunur:
        //     GPIO18 dusuk = C28x oznitelik cikariyor
        //     GPIO67 dusuk = CLA inference yapiyor   (Cla1Task1 suruyor)
        // Sonuc o modda bir pencere gecikmeli kaydedilir (bkz. run_inference).
        //
        // TEK TAMPON, VE NEDEN YETIYOR: CLA ile CPU'nun paylastigi tek
        // degistirilebilir tampon input_image[]. workspace[] ve fResult[]
        // yalnizca CLA'nin, agirliklar sabit, ve CPU fResult'i yalnizca
        // cla_done 1 iken okuyor.
        //
        // input_image[] pencerenin SONUNDA yaziliyor (mfcc_extract bittikten
        // sonraki int16 donusum dongusu). CLA hemen ardindan tetiklendigi
        // icin, bir sonraki yazmaya kadar elinde TAM BIR oznitelik cikarma
        // periyodu var. Kosul bu kadar basit:
        //
        //     inference <= oznitelik cikarimi
        //      34 ms    <= 202 ms        -> 168 ms pay  (4 kanal, olculen)
        //
        // Yani eskiden "verimlilik" sanilan esik, tek tamponlu bu tasarimin
        // DOGRULUK siniri. Asilirsa kod cokmez, hata da vermez: CLA hala
        // onceki pencereyi okurken CPU ustune yeni pencereyi yazar ve sonuc
        // sessizce bozulur. O noktada input_image ping-pong tamponlanmali.
        run_inference(golden_cycle_idx);
        GPIO_WritePin(19, 1);

        golden_cycle_idx++;
        if (golden_cycle_idx >= NUM_CLASSES) golden_cycle_idx = 0;

//        CLA_runTest();

//        if(done == 1)
//                {
//                    static uint32_t f,o;
//
//                    // DİKKAT: golden_adc YERİNE adcData0 KULLANILIYOR
//                    mfcc_extract((const unsigned int*)adcData0, features);
//
//                    int idx = 0;
//                    for ( f = 0; f < FEAT_N_FRAMES; f++) {
//                        for ( o = 0; o < FEAT_N_OUT; o++) {
//                            float scaled = features[f][o] * INPUT_SCALE;
//                            if (scaled > 32767.0f) scaled = 32767.0f;
//                            if (scaled < -32768.0f) scaled = -32768.0f;
//                            input_image[idx++] = (int16_t)scaled;
//                        }
//                    }
//
//                    convolution_int16(input_image, workspace, conv1_weights, conv1_bias, IMAGE_H,  IMAGE_W, CONV1_WEIGHT_SCALE);
//                    relu_activation_int16( workspace,  MAX_ELEMENTS);
//                    max_pooling_2x1_int16(workspace, &workspace[CONV1_SIZE], CONV1_OUT_H, CONV1_OUT_W, NUM_FILTERS);
//
//                    convolution_2_int16(&workspace[CONV1_SIZE], &workspace[CONV1_SIZE + POOL1_SIZE], conv2_weights, conv2_bias, POOL1_OUT_H, POOL1_OUT_W, CONV2_WEIGHT_SCALE);
//                    relu_activation_int16(&workspace[CONV1_SIZE + POOL1_SIZE], CONV2_SIZE);
//                    max_pooling_2x1_int16(&workspace[CONV1_SIZE + POOL1_SIZE], input_vector, CONV2_OUT_H, CONV2_OUT_W, NUM_FILTERS);
//
//                    // CLA işlemini tetikle
//                    CLA_runTest();
//
//                    // 3. EKSİK: BAYRAĞI SIFIRLA
//                    // İşlemler bitti. DMA'nın bir sonraki 1024'ü doldurmasını beklemek için bayrağı indir.
//                    done = 0;
//                }

     }

}



// fResult[]'in argmax'i -> g_predicted_class, g_logits[], dogruluk sayaclari.
static void record_result(uint16_t expected_class)
{
    uint16_t c, best = 0;
    for (c = 0; c < NUM_CLASSES; c++) {
        g_logits[c] = fResult[c];
        if (fResult[c] > fResult[best]) best = c;
    }
    g_predicted_class = best;
    g_predicted_name = golden_names[best];
    g_total_count++;
    if (best == expected_class) g_correct_count++;
}

// INFERENCE: input_image[] (mfcc_extract'in int16 ciktisi) ->
//   S1 conv1 + relu + pool1 -> S2 conv2 + relu + pool2 -> S3 dense
//   -> fResult[] -> sinif karari (record_result).
//
// Her asama cla_pipeline_config.h'a gore ya convolution.c'deki CPU
// fonksiyonuyla, ya da dense_layer.cla'daki CLA goreviyle kosar. Uc bayrak da 0
// iken (tamamen CPU) butun ag bu fonksiyonun icinde, C28x'te calisir ve
// fonksiyon dondugunde fResult[] hazirdir.
void run_inference(uint16_t expected_class)
{
#if CLA_SINGLE_TASK
    // Uc asama da CLA'da -- paralellik modu: tek tetik, asenkron, CLA
    // beklenmez. Bir onceki tetigin sonucu, CLA bitirdiyse, ait oldugu
    // pencerenin sinifiyla kaydedilir -- yeni tetik fResult'i yeniden
    // yazmadan ONCE.
    static uint16_t pending = 0, pending_class = 0;
    if (pending && cla_done) {
        record_result(pending_class);
    }
    cla_done = 0;
    Cla1ForceTask1();              // conv1+conv2+dense tek CLA gorevinde
    pending = 1;
    pending_class = expected_class;
#else
    cla_done = 0;

    // S1: conv1 + relu + pool1  (input_image -> workspace[0..POOL1_SIZE))
  #if CLA_STAGE_CONV1_POOL1
    Cla1ForceTask1andWait();
  #else
    conv1_pool_fused_cpu(input_image, workspace, conv1_weights, conv1_bias);
  #endif

    // S2: conv2 + relu + pool2  (workspace -> workspace[POOL1_SIZE..])
  #if CLA_STAGE_CONV2_POOL2
    Cla1ForceTask2andWait();
  #else
    conv2_pool_fused_cpu(workspace, &workspace[POOL1_SIZE], conv2_weights, conv2_bias);
  #endif

    // S3: dense / FC  (pool2 ciktisi -> fResult[NUM_CLASSES])
  #if CLA_STAGE_DENSE
    Cla1ForceTask3andWait();
  #else
    dense_fc_cpu(&workspace[POOL1_SIZE], weight, fc_bias, fResult);
  #endif

    cla_done = 1;
    record_result(expected_class);
#endif
}
//
////
// CLA_configClaMemory - Configure the CLA memory
//
void CLA_configClaMemory(void)
{
    extern uint32_t Cla1funcsRunStart, Cla1funcsLoadStart, Cla1funcsLoadSize;
    extern uint32_t Cla1DataRunStart, Cla1DataLoadStart, Cla1DataLoadSize;
    EALLOW;

#ifdef _FLASH
    //
    // Copy over code from FLASH to RAM
    //
    memcpy((uint32_t *)&Cla1funcsRunStart, (uint32_t *)&Cla1funcsLoadStart,
           (uint32_t)&Cla1funcsLoadSize);

    // DUZELTME (soguk acilis): CLA'nin okudugu const agirliklar (CLADataLS1:
    // weight[], conv agirlik/bias, fc_bias) da flash'tan RAM'e kopyalanmali --
    // eskiden bunu sadece debugger yapiyordu, kart kendi basina acilinca CLA
    // sifir/copp agirliklarla calisiyordu (bkz. linker cmd'deki not). Bu
    // fonksiyon herhangi bir CLA gorevi tetiklenmeden once cagriliyor.
    memcpy((uint32_t *)&Cla1DataRunStart, (uint32_t *)&Cla1DataLoadStart,
           (uint32_t)&Cla1DataLoadSize);
#endif //_FLASH

    //
    // Initialize and wait for CLA1ToCPUMsgRAM
    //
    MemCfgRegs.MSGxINIT.bit.INIT_CLA1TOCPU = 1;
    while(MemCfgRegs.MSGxINITDONE.bit.INITDONE_CLA1TOCPU != 1){};

    //
    // Initialize and wait for CPUToCLA1MsgRAM
    //
    MemCfgRegs.MSGxINIT.bit.INIT_CPUTOCLA1 = 1;
    while(MemCfgRegs.MSGxINITDONE.bit.INITDONE_CPUTOCLA1 != 1){};

    //
    // Select LS4RAM and LS5RAM to be the programming space for the CLA
    // First configure the CLA to be the master for LS4 and LS5 and then
    // set the space to be a program block
    //
//    MemCfgRegs.LSxMSEL.bit.MSEL_LS4 = 1;
//    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS4 = 1; // program memory
    MemCfgRegs.LSxMSEL.bit.MSEL_LS5 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS5 = 1;

    //
    //Next configure LS0RAM and LS1RAM as data spaces for the CLA
    // First configure the CLA to be the master for LS0(1) and then
    // set the spaces to be code blocks
    //
    MemCfgRegs.LSxMSEL.bit.MSEL_LS0 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS0 = 0; // data memory


    MemCfgRegs.LSxMSEL.bit.MSEL_LS1 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS1 = 0;


    MemCfgRegs.LSxMSEL.bit.MSEL_LS2 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS2 = 0;

    MemCfgRegs.LSxMSEL.bit.MSEL_LS3 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS3 = 0;

    MemCfgRegs.LSxMSEL.bit.MSEL_LS4 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS4 = 0;
    EDIS;
}

//
// CLA_initCpu1Cla1 - Initialize CLA1 task vectors and end of task interrupts
//
void CLA_initCpu1Cla1(void)
{
    //
    // Compute all CLA task vectors
    // On Type-1 CLAs the MVECT registers accept full 16-bit task addresses as
    // opposed to offsets used on older Type-0 CLAs
    //
    EALLOW;
    // GUNCELLEME (asama bazli bolusum): yalnizca DERLENEN gorevlerin vektoru
    // yazilir. Tek gorev modunda (uc asama da CLA'da) sadece MVECT1; karma
    // dagilimda asama basina MVECT1=S1, MVECT2=S2, MVECT3=S3. Secilmeyen
    // asamanin gorevi hic derlenmedigi icin adresi de alinamaz -- bu yuzden
    // kosullu (bkz. dense_layer.cla).
#if CLA_SINGLE_TASK
    Cla1Regs.MVECT1 = (uint16_t)(&Cla1Task1);
#else
  #if CLA_STAGE_CONV1_POOL1
    Cla1Regs.MVECT1 = (uint16_t)(&Cla1Task1);
  #endif
  #if CLA_STAGE_CONV2_POOL2
    Cla1Regs.MVECT2 = (uint16_t)(&Cla1Task2);
  #endif
  #if CLA_STAGE_DENSE
    Cla1Regs.MVECT3 = (uint16_t)(&Cla1Task3);
  #endif
#endif

    Cla1Regs.MCTL.bit.IACKE = 1;
    Cla1Regs.MIER.all = 0x00FF;

    PieVectTable.CLA1_1_INT  = &cla1Isr1;

    //
    PieCtrlRegs.PIEIER11.bit.INTx1 = 1;
    IER |= (M_INT11 );
    EDIS;
}

////
//// cla1Isr1 - CLA1 ISR 1
////
__interrupt void cla1Isr1 ()
{
    // EKLENDI: Cla1Task1 (conv1+conv2+FC, tamami) bitince buraya dusuluyor --
    // main loop artik bunu bekleyebiliyor.
    cla_done = 1;

    PieCtrlRegs.PIEACK.all = M_INT11;


}
