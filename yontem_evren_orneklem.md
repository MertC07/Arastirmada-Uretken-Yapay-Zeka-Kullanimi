# 4.2 Evren ve Örneklem (Veri Seti ve Sınıflandırma)

## 4.2.1 Araştırmanın Evreni

Nicel araştırmalarda evren, araştırma bulgularının genellenmek istendiği ya da üzerinde çıkarım yapılmayı hedeflenen bütünün tamamını ifade etmektedir (Fraenkel ve ark., 2012). Bu araştırmanın evreni; karayolu ortamında faaliyet gösteren motorlu taşıtları, bu taşıtları kullanan sürücüleri ve yol üzerindeki savunmasız yol kullanıcılarını (yaya, bisikletli) kapsayan gerçek dünya trafik sahnelerinin tümünü oluşturmaktadır. Söz konusu evren; farklı coğrafi bölgeleri, kentsel ve kırsal yol koşullarını, günün çeşitli saatlerini ve olası hava koşullarını bünyesinde barındıran geniş ve heterojen bir yapıya sahiptir. Bu nedenle evrenin bütününe erişim, pratik ve teknik kısıtlar nedeniyle mümkün olmamakta; araştırmanın bulguları, belirlenen örneklem çerçevesinde yorumlanmaktadır.

## 4.2.2 Örneklem ve Veri Setinin Oluşturulması

Araştırmanın örneklemi, derin öğrenme modeli eğitimi ve test süreçleri için oluşturulan **etiketlenmiş görüntü veri setinden** meydana gelmektedir. Veri seti, nesne tespiti ve bilgisayarlı görü alanında yaygın biçimde kullanılan **Roboflow** platformu aracılığıyla derlenmiş ve etiketleme (annotation) işlemleri bu platform üzerinde gerçekleştirilmiştir.

Veri setinin büyüklüğü ve içerik çeşitliliği, bilgisayarlı görü ve nesne tespiti literatüründen çıkarım yapılarak belirlenmiştir. Bu çerçevede; bir modelin genelleştirilebilir bir temsil öğrenebilmesi için gereken minimum veri büyüklüğüne ilişkin yayımlanmış kılavuz değerler (Laroca ve ark., 2021; Qiu ve ark., 2024) ile sınıf dengesizliğinin model performansı üzerindeki olumsuz etkisini azaltmaya yönelik öneriler (Arya ve ark., 2022) dikkate alınmıştır.

### Sınıf Yapısı ve Etiketleme Protokolü

Veri seti, araştırmanın öncelikli tespit hedefi olan **emniyet kemeri** (seatbelt) sınıfı ekseninde yapılandırılmıştır. Etiketleme protokolü; her görüntüde kemer şeridinin omuzdan bel hizasına uzanan görünür kısmının sıkı sınırlayıcı kutu (tight bounding box) ile işaretlenmesi ilkesine dayandırılmıştır. Tutarsız etiketlemenin model başarımı üzerindeki bozucu etkisini en aza indirmek amacıyla, araştırma ekibi tarafından yazılı bir etiketleme kılavuzu hazırlanmış ve tüm ekip üyelerinin bu kılavuza bağlı kalarak çalışması sağlanmıştır. Roboflow platformunun sağladığı merkezi proje yönetimi altyapısı, ekip üyelerinin veri katkılarının tek bir proje kapsamında birleştirilmesine ve etiket kalitesinin merkezî olarak denetlenmesine olanak tanımıştır.

## 4.2.3 Örneklem Çeşitliliği ve Temsil Gücü

Nesne tespiti modellerinin gerçek dünya koşullarındaki genellenebilirliği, büyük ölçüde eğitim verisinin çeşitliliğiyle doğru orantılıdır (Sami ve ark., 2024; Zabłocki ve ark., 2024). Bu bağlamda veri setinin temsil gücünü artırmak amacıyla görüntü toplama ve seçim sürecinde aşağıdaki çeşitlilik boyutları sistematik olarak gözetilmiştir:

**Işık ve Zaman Koşulları:** Veri setine gündüz (tam aydınlık), alacakaranlık ve gece saatlerine ait görüntüler dahil edilmiştir. Bu ayrım; gerçek trafik gözetleme sistemlerinin günün her saatinde kesintisiz çalışması zorunluluğundan kaynaklanmakta ve düşük ışık koşullarına duyarlılığın ölçülmesine imkân tanımaktadır.

**Hava Koşulları:** Açık hava, bulutlu ve yağışlı koşulları temsil eden görüntüler derlenerek modelin çevresel değişkenlere karşı dayanıklılığı veri düzeyinde güvence altına alınmıştır. Literatür, olumsuz hava koşullarının tespit doğruluğunu anlamlı ölçüde düşürebildiğini ortaya koymaktadır (Arya ve ark., 2022).

**Araç ve Yolcu Çeşitliliği:** Farklı araç tiplerini (binek otomobil, hafif ticari araç) ve farklı fiziksel görünüme sahip sürücüleri kapsayan görüntüler bir arada kullanılmıştır. Bu çeşitlilik, modelin belirli bir araç tipine veya sürücü profiline aşırı uyum (overfitting) geliştirmesinin önüne geçmeyi hedeflemektedir.

**Kamera Açısı ve Mesafe:** Görüntüler farklı çekim mesafeleri ve açıları gözetilerek derlenmiştir. Gerçek yol denetim senaryolarında kamera konumunun değişkenlik göstereceği varsayımı, bu çeşitliliğin temel gerekçesini oluşturmaktadır.

## 4.2.4 Veri Artırma (Data Augmentation)

Ham görüntü sayısının model genelleştirilmesi için gereken eşiğin altında kalma riskini azaltmak ve veri çeşitliliğini yapay yollarla zenginleştirmek amacıyla **veri artırma (data augmentation)** teknikleri uygulanmıştır. Artırma işlemi yalnızca eğitim kümesine uygulanmış; doğrulama ve test kümeleri gerçek görüntülerden oluşan özgün hâlleri korunarak kullanılmıştır. Bu yaklaşım, modelin yalnızca artırılmış veriye değil, ham görüntülere ne denli iyi genelleyebildiğinin nesnel biçimde ölçülmesine olanak tanımaktadır.

Uygulanan artırma dönüşümleri şunlardır:

| Artırma Türü | Parametre Aralığı | Gerekçe |
|---|---|---|
| Parlaklık (Brightness) | ±%25 | Farklı aydınlatma koşullarını simüle etmek |
| Pozlama (Exposure) | ±%15 | Kamera sensörü çeşitliliğini temsil etmek |
| Bulanıklaştırma (Blur) | Maks. 1,5 px | Hareket bulanıklığı ve odak kaymasını temsil etmek |
| Gürültü (Noise) | Maks. %2 | Düşük kaliteli kamera görüntülerini simüle etmek |
| Döndürme (Rotation) | ±5° | Araç eğimi ve kamera açı sapmalarını temsil etmek |

Emniyet kemeri şeridinin aracın sürücü tarafında belirli bir açıyla konumlandığı göz önünde bulundurularak yatay çevirme (horizontal flip) dönüşümü kasıtlı olarak devre dışı bırakılmıştır; bu dönüşümün uygulanması durumunda direksiyon simidinin ve kemer konumunun gerçekle uyumsuz bir aynalama içereceği değerlendirilmiştir. Her eğitim görüntüsü için **3 artırılmış çıktı** üretilmiş; böylece ham görüntü başına toplam veri katkısı üç katına çıkarılmıştır.

## 4.2.5 Eğitim, Doğrulama ve Test Kümelerine Bölünme

Veri setinin modeli hem başarılı biçimde öğretilebilmesi hem de tarafsız bir zemin üzerinde değerlendirilebilmesi için üç alt kümeye ayrılması gerekmektedir. Bu gereklilik; eğitim, doğrulama ve test kümelerinin işlevsel bağımsızlığının sağlanması ilkesine dayanmakta ve derin öğrenme literatüründe yaygın bir uygulama standardı olarak kabul görmektedir (Goodfellow ve ark., 2016). Araştırmada veri seti aşağıdaki oranlarda bölünmüştür:

| Küme | Oran | İşlev |
|---|---|---|
| **Eğitim (Train)** | %70 | Model ağırlıklarının optimize edilmesi |
| **Doğrulama (Validation)** | %20 | Eğitim süreci boyunca aşırı uyumun izlenmesi ve hiperparametre ayarı |
| **Test** | %10 | Modelin daha önce hiç görmediği verilerle nihai performans değerlendirmesi |

Test kümesi; tüm eğitim süreci boyunca modelden tamamen yalıtılmış, yalnızca nihai performans raporlaması aşamasında kullanılmıştır. Bu yaklaşım, test setine yönelik bilgi sızıntısının (data leakage) önlenmesi ve elde edilen performans metriklerinin nesnel geçerliliğinin korunması bakımından kritik öneme sahiptir.

---

### Kaynakça (Evren ve Örneklem Bölümü)

Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2022). Deep learning-based road damage detection and classification for multiple countries. *Automation in Construction*, *132*, 103935. https://doi.org/10.1016/j.autcon.2021.103935

Fraenkel, J. R., Wallen, N. E., & Hyun, H. H. (2012). *How to design and evaluate research in education* (8. bs.). McGraw-Hill.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Laroca, R., Zanlorensi, L. A., Gonçalves, G. R., Todt, E., Schwartz, W. R., & Menotti, D. (2021). An efficient and layout-independent automatic license plate recognition system based on the YOLO detector. *IET Intelligent Transport Systems*, *15*(4), 483–503. https://doi.org/10.1049/itr2.12030

Qiu, L., Rao, J., & Zhao, X. (2024). Seatbelt detection algorithm improved with lightweight approach and attention mechanism. *Applied Sciences*, *14*(8), 3346. https://doi.org/10.3390/app14083346

Sami, H., ve ark. (2024). *A comprehensive review on artificial intelligence empowered solutions for enhancing pedestrian and cyclist safety* [Preprint]. arXiv. https://arxiv.org/abs/2510.03314

Zabłocki, M., ve ark. (2024). Utilizing YOLOv8 for enhanced traffic monitoring in intelligent transportation systems (ITS) applications. *Digital Signal Processing*. https://doi.org/10.1016/j.dsp.2024.104481
