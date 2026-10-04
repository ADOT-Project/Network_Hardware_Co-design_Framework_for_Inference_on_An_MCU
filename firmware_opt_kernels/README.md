# Hızlandırılmış çıkarım çekirdekleri (deneme)

`CNN_Inference_Edge-AI_with_CLA` içindeki üç aşamanın (conv1+pool1,
conv2+pool2, dense) işaretçili ve 3×3'ü açılmış sürümleri. Amaç: `-O0`
derlemede CLA ile C28x arasındaki farkı daraltmak.

Bu klasör CCS projesinin **dışında** duruyor. CCS, proje klasörünün altındaki
her kaynak dosyayı derlemeye kattığı için içine konsaydı aynı fonksiyonlar iki
kez tanımlanır ve `Cla1Prog` taşardı.

```
drop_in/    projeye kopyalanacak dört dosya
original/   eski convolution.c, dense_layer.cla, mfcc_extract.c (geri dönüş + test referansı)
pc_test/    eski ve yeni çekirdekleri PC'de karşılaştıran test
```

## Ne değişti

1. Çarpma başına indis hesabı yok; iki işlenen de ilerleyen işaretçiden okunuyor.
2. 3×3 pencere düz yazıldı: 9 çarpma, arada döngü ve dallanma yok.
3. Bias ölçeklemesi pencere başına değil filtre başına bir kez.
4. ReLU ve 2×1 havuzlama float toplamlar üzerinde; ölçek bölmesi ve int16
   dönüşümü çıkış başına bir kez (eskiden iki).
5. CPU ve CLA gövdeleri tek dosyadan (`fused_kernels_opt.h`) geliyor; iki
   çekirdeğin farklı şey hesaplaması artık mümkün değil.

Dense katman da işaretçili, ama toplam işin %1'i olduğu için süreye etkisi yok.

## PC doğrulaması

```bash
cd pc_test
./run.sh                             # varsayılan ayarlar
./run.sh -DKERNEL_ADDR_WALK=0        # sabit ofsetli adresleme
./run.sh -DKERNEL_POOL_FLOAT=0       # havuzlama int16 üzerinde (eski sıra)
./run.sh -DKERNEL_USE_RECIPROCAL=1   # bölme yerine 1/SCALE ile çarpma
```

Her çalıştırma 8 CPU/CLA bölme düzeninin hepsini, düzen başına 300 çalışma
zamanı girişi ve 300 taşma girişiyle dener (9 kanal, 2 sınıf, 65×12 model).
2026-10-03 sonuçları:

| Ayar | Çalışma zamanı girişleri | int16 taşan girişler |
|---|---|---|
| varsayılan | bit düzeyinde aynı | farklı |
| `KERNEL_ADDR_WALK=0` | bit düzeyinde aynı | farklı |
| `KERNEL_POOL_FLOAT=0` | bit düzeyinde aynı | bit düzeyinde aynı |
| `KERNEL_USE_RECIPROCAL=1` | 1.139.400 aktivasyonun 13'ü 1 LSB farklı; logit'lerde en çok %0,24; karar değişmiyor | farklı |

"Taşan girişler": aktivasyon int16'ya sığmadığında eski çekirdekler de yanlış
sonuç veriyor; iki sıra yalnızca farklı yanlış değeri seçiyor. Bu durumda bile
aynı kalsın isterseniz `KERNEL_POOL_FLOAT=0`.

PC testi sayıların aynı olduğunu gösterir, **hızı göstermez**. Süre yalnızca
kartta ölçülür.

## Projeye alma

1. `drop_in/` içindeki dört dosyayı `CNN_Inference_Edge-AI_with_CLA/` içine
   kopyalayın (`convolution.c` ve `dense_layer.cla` üzerine yazılır,
   `mfcc_extract.c` üzerine yazılır, `fused_kernels_opt.h` yeni).
2. Derleyin ve `Debug/dense_layer_cla.map` içinde `Cla1Prog` boyutuna bakın
   (sınır 0x800 = 2048 kelime; eski çekirdekler 2042 tutuyordu).
3. `tools/board_golden_test/run_board_golden_test.sh 111 000` ile kartta
   doğrulayın. Varsayılan ayarlarda `expected_logits.txt` değişmez.
4. Süreyi GPIO67 ile ölçün; aynı model için eski ve yeni, CPU ve CLA: dört değer.

Geri dönmek için `original/` içindeki üç dosyayı geri kopyalayıp
`fused_kernels_opt.h` dosyasını silin.

## Ayarlar

`fused_kernels_opt.h` başında ya da CCS'te *Predefined Symbols* ile. C28x ve
CLA derleyicisine **aynı** değer verilmeli, yoksa iki çekirdek farklı hesaplar.

| Ayar | Varsayılan | Ne zaman değiştirilir |
|---|---|---|
| `KERNEL_ADDR_WALK` | 1 | 0'ı da ölçün; hangisi hızlıysa onu tutun. Sonuç aynı. |
| `KERNEL_POOL_FLOAT` | 1 | Taşma durumunda da eskiyle aynı sonuç gerekiyorsa 0. |
| `KERNEL_USE_RECIPROCAL` | 0 | CLA'da yazılım bölmesini kaldırmak için 1. Bit düzeyinde aynı değildir; `expected_logits.txt` yeniden üretilmeli. |

## Cla1Prog sığmazsa

Açılmış 3×3 kod boyutunu büyütür; sığıp sığmadığı derlenmeden bilinemez.
Link `Cla1Prog` için yer hatası verirse, sırayla:

1. `KERNEL_ADDR_WALK` değerini değiştirip yeniden deneyin; iki biçimin kod
   boyutu farklıdır.
2. Dense aşamasını CPU'ya alın (`CLA_STAGE_DENSE 0`). Süreye etkisi yok
   denecek kadar az, ama CPU artık CLA'yı beklemek zorunda kalır; paralellik
   için geçici bir ölçüm düzeni olarak kullanın.
3. LS4'ü de CLA programına verin: linker dosyasında veri havuzunu
   `RAMLS_0_1_2_3` (0x2000) yapıp program için `RAMLS4_5` (0x1000) tanımlayın
   ve `main.c` içindeki CLA bellek ayarında LS4'ü program olarak işaretleyin.
   Bedeli: CLA veri havuzu 10.240 kelimeden 8.192'ye iner; 65×12 giriş ve
   2 sınıfta sığan en büyük ağ 12 kanaldan 9 kanala düşer.

## MFCC (`mfcc_extract.c`)

Aynı öznitelikler, farklı hesap sırası: gerçek girişe özel yarı boyutlu FFT
(512 noktalı karmaşık FFT yerine 256 noktalı), 4'lü taban (8 geçiş yerine 4),
ilk geçişte pencere + paketleme + bit çevirme + ilk iki FFT aşaması birlikte,
aşama sırasına göre dizilmiş hazır katsayı tabloları, işaretçili döngüler.
Dilimin float kopyası (`x[]`, 33.792 kelime RAM) kaldırıldı.

```bash
cd pc_test && ./run_mfcc.sh
```

Golden dilimlerde ve aynı kayıtların kaydırılmış hâllerinde (2026-10-03):

| | Sonuç |
|---|---|
| Eski ile yeni öznitelik farkı | en çok 9,6e-4 (z-skor birimi) |
| Eğitim referansına (`golden_expected`) uzaklık | eski 1,1e-4 – 2,3e-4, yeni 7e-6 – 2,2e-5 |
| Değişen ağ girişi (int16) | 780'de 0–29, hepsi 1 LSB |
| Logit farkı | en çok %0,026 |
| Karar | her dilimde aynı |

Bit düzeyinde aynı değildir (farklı FFT sırası); yeni sürüm eğitim
referansına daha yakındır. Tablolar **ilk çağrıda** kurulur (yaklaşık bin
`cosf`/`sinf`), bu yüzden ilk pencere yavaştır: süreyi ikinci pencereden itibaren
ölçün.

`10*log10` artık kütüphanedeki `log10f` ile değil, dosyadaki `power_db()` ile
hesaplanıyor (float'ın üssü + mantis için kısa seri; tam değere göre en çok
1e-5 dB sapma, float'ın 100 dB civarındaki çözünürlüğü kadar). Eski davranış
için `MFCC_FAST_LOG=0`; ikisini ölçerek `log10f`'in payını görebilirsiniz.

## Sonrasında

- ADOT'un zamanlama modeli eski çekirdeklere göre kalibre edildi (MAC başına
  133 çevrim CLA, 71 çevrim C28x). Yeni çekirdeklerle bu değerler ve −%4/+%13
  bandı geçersiz; kart ölçümüyle yeniden bağlanmalı.
- `weights.h` içindeki `input_image` başlangıç örneği ±32767'ye ölçekli.
  Çalışma zamanında ağa giren değerler (öznitelik × 256) bunun yaklaşık kırkta
  biri. Bu örnekle 9 kanallı modelde conv1 çıkışında 38, conv2 çıkışında 2
  pencere int16'yı aşıyor. Kart testi bu diziyi değil golden vektörleri
  kullandığı için sonuçları etkilemez, ama diziyi doğrudan besleyen bir deneme
  yanlış logit üretir.
