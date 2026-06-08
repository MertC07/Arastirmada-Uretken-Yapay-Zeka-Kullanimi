# Hakem Değerlendirmesi

**Makale:** Yapay Zeka ve 5G Entegrasyonuyla Akıllı Yol Güvenliği: Gerçek Zamanlı Nesne Tespiti ve Sürücü Davranış Analizi Üzerine Betimsel Bir Araştırma

**Değerlendirme Türü:** Çift Kör Hakem İncelemesi (Double-Blind Peer Review)

**Karar:** Revizyon Önerilir *(Major Revision)*

---

## I. Güçlü Yönler

### G-1. Bütünleşik Sistem Perspektifi

Literatürdeki bileşen odaklı çalışmaların en büyük kör noktasını — birden fazla güvenlik modülünün aynı anda çalıştırılması — doğrudan ele alması, çalışmanın en özgün yanıdır. Pipeline + Strategy tasarım desenleri bu bütünleşik yaklaşımın metodolojik omurgasını tutarlı biçimde oluşturmaktadır.

### G-2. Güçlü ve Tutarlı Literatür Sentezi

18 çalışmanın tematik alt başlıklar altında sistematik biçimde özetlenmesi ve her birinin araştırmanın boşluklarıyla ilişkilendirilmesi akademik olgunluk göstergesidir. Literatür değerlendirmesi (Bölüm 2.8) özellikle başarılıdır; altı temel eksenin tanımlanması ve her birindeki boşlukların net biçimde ortaya konması takdire değerdir.

### G-3. Somut Nicel Bulgular

YOLOv11l'in plaka tespitinde 0,986 mAP@50, RT-DETR-L'nin telefon tespitinde 1,000 Precision, YOLOv11n+αSiLU'nun esneme tespitinde 0,995 mAP@50 gibi somut metrikler tartışmayı soyut olmaktan kurtarmakta ve literatürle kıyaslamayı mümkün kılmaktadır.

### G-4. αSiLU Bulgusunun Özgünlüğü

Nano mimaride αSiLU aktivasyon fonksiyonunun beklenmedik başarımı, literatürde henüz sistematik biçimde belgelenmemiş özgün bir bulgu olarak öne çıkmaktadır. Bu bulgu, özellikle kenar hesaplama bağlamında ileri araştırmalar için anlamlı bir başlangıç noktası sunmaktadır.

### G-5. Veri Artırma Kararlarının Gerekçelendirilmesi

Yatay çevirmenin (horizontal flip) devre dışı bırakılması ve bunun emniyet kemeri geometrisiyle ilişkilendirilmesi gibi metodolojik kararların gerekçeli biçimde açıklanması, yöntem bölümünün güvenilirliğini artırmaktadır.

### G-6. Türkiye Bağlamının Etkin Kullanımı

TÜİK 2023 verilerinin (6.548 ölü, %25,2 artış) hem giriş hem de tartışmaya entegre edilmesi, araştırmanın yerel politika bağlamıyla teorik katkısını başarılı biçimde bağdaştırmaktadır.

### G-7. Araştırma Sorularının Mantıksal Hiyerarşisi

Beş araştırma sorusunun *"ne kadar iyi → 5G ne katar → birlikte ne olur → zorlu koşullarda dayanıklı mı → politika engelleri"* şeklinde katmanlı bir yapı oluşturması, okuyucunun araştırmanın niyetini kolayca kavramasını sağlamaktadır.

---

## II. Zayıf Yönler

### A. Yapısal / Metodolojik Sorunlar

#### Z-1. Bulgular Bölümünün Yokluğu *(En Kritik Eksiklik)*

Makale, tartışmayı bulgulara dayandırdığını söylemekte ancak ayrı bir Bulgular bölümü sunmamaktadır. Tartışmada yer alan sayısal değerler (mAP, Precision, FPS) herhangi bir tabloya, şekile veya sistematik bir bulgu sunumuna atıfta bulunmaksızın doğrudan metne gömülmüştür. Hakem, hangi verinin deneysel sonuç, hangi verinin literatür karşılaştırması olduğunu ayrıştırmakta güçlük çekmektedir.

#### Z-2. Veri Seti Boyutunun Belirsizliği

Yöntem bölümü etiketleme protokolünü, veri artırma parametrelerini ve bölünme oranlarını ayrıntılı biçimde açıklarken **ham görüntü sayısını hiç belirtmemektedir.** "~200 görüntü × 3 artırma = ~600 eğitim örneği" gibi bir bilgi, çalışmanın tekrarlanabilirliği ve bulguların güvenilirliği için zorunludur. Bu bilgi olmaksızın ulaşılan doğruluk değerlerinin ne kadar güçlü kanıt taşıdığı değerlendirilemez.

#### Z-3. 5G'nin Teorik Varsayım Olmaktan Öteye Geçememesi

AS-2 "5G ve Kenar Hesaplama Entegrasyonu" olarak adlandırılmış ancak çalışmada gerçek bir 5G testi yapılmamıştır. *"6,3 ms işleme süresi + 1 ms 5G gecikmesi = teorik <10 ms"* çıkarımı spekülatiftir ve Herbst ve ark. (2021)'in farklı bir sistem üzerinde yaptığı testlere dayanmaktadır. Bu durum AS-2'yi yanıtlanmamış bırakmakta; başlık ile içerik arasında ciddi bir tutarsızlık oluşturmaktadır.

#### Z-4. Eğitim Hiperparametrelerinin Raporlanmaması

Epoch sayısı, batch size, öğrenme oranı, optimizatör türü ve donanım konfigürasyonu (GPU modeli, VRAM) gibi temel eğitim parametreleri hiçbir bölümde verilmemiştir. Bu bilgisizlik, çalışmanın tekrar edilebilirliğini ciddi biçimde zedelemektedir.

#### Z-5. Betimsel Desen ile Karşılaştırmalı Bulgular Arasındaki Gerilim

Çalışma betimsel bir desen benimsediğini açıklamakta, ancak ardından birden fazla modeli aynı veri seti üzerinde karşılaştırmaktadır. Karşılaştırmalı model değerlendirmesi teknik olarak deneysel bir prosedürdür. "Betimsel" etiketinin bu tasarımı tam karşılayıp karşılamadığı ya da *"karşılaştırmalı değerlendirme"* gibi daha uygun bir alt desen tanımının yapılması gerekip gerekmediği netleştirilmelidir.

#### Z-6. Modeller Arası Adil Karşılaştırma Koşullarının Belirsizliği

Farklı mimariler (YOLOv11l, RT-DETR-L, YOLOv8n) karşılaştırılmakta, ancak bu modellerin aynı donanım üzerinde, aynı epoch sayısıyla ve aynı veri bölünmesiyle eğitilip eğitilmediği belirtilmemektedir. Farklı koşullarda eğitilmiş modeller arasındaki kıyaslama yanıltıcı sonuçlar doğurabilir.

---

### B. İçerik Eksiklikleri

#### Z-7. Hız Tahmini Modülüne Hiç Değinilmemesi

Araştırma soruları ve kapsam kısmında "hız ihlali" parametresi kapsam dahilinde tanımlanmakta, ancak hız tahmini modülü ne yöntem ne de bulgularda işlenmektedir. Bu tutarsızlık okuyucuda yapısal bir boşluk yaratmaktadır.

#### Z-8. Sınıflar Arası Denge Tablosunun Yokluğu

`no_seatbelt` sınıfındaki dramatik düşüş (0,337 mAP@50) anlamlı bir bulgu olarak tartışılmakta, ancak veri setinde bu sınıfa kaç örnek düştüğü ve pozitif/negatif örnek oranının ne olduğu açıklanmamaktadır. Okuyucu, düşük performansın gerçekten mimari bir sınırlılıktan mı yoksa salt veri dengesizliğinden mi kaynaklandığını ayırt edememektedir.

#### Z-9. Güvenilirlik ve Geçerlik Tartışmasının Eksikliği

Inter-annotator agreement (etiketleyiciler arası uyum) hesaplanmamış, etiket tutarlılığını ölçen herhangi bir metrik sunulmamıştır. Dört kişinin ayrı ayrı etiketlediği bir veri setinde bu ölçüm metodolojik bir zorunluluktur.

---

### C. Yazım / Sunum Sorunları

#### Z-10. Kaynakça Tutarsızlıkları

- "PMC, 2024" atfı hem giriş metninde hem de kaynak listesinde eksik veya belirsizdir.
- Rahutomo ve ark. (2024) için DOI yer tutucusu (`000XX`) hâlâ doldurulmamıştır.
- García-González ve ark. (2025) DOI'si muhtemelen yerleştirilmiş bir yer tutucudur; doğrulanması gerekmektedir.
- Bazı "ve ark." atamaları tam yazar bilgisi içermemektedir; APA 7 standardı üç veya daha fazla yazarlı atıflarda ilk atıfta tüm yazarların, sonrakilerde "ve ark." kullanımını zorunlu kılmaktadır.

#### Z-11. Bayrak Emoji Kullanımı

İlgili çalışmalar bölümündeki ülke bayrakları (🇨🇳, 🇹🇷, 🇪🇸 vb.) akademik bir dergi makalesinde standart dışı görünmektedir. Yazar ülkeleri parantez içinde metin olarak belirtmelidir.

---

## III. Sınırlılıkları Güçlendirme Önerileri

---

### Ö-1. Bulgular Bölümü Ekleyin *(Öncelikli)*

Tartışmadan önce gelen bağımsız bir Bulgular bölümü oluşturun. Her görev için tüm modellerin mAP@50, mAP@0.5:0.95, Precision, Recall ve FPS değerlerini içeren **karşılaştırmalı bir tablo** sunun. Bu tablo, tartışmanın dayandığı kanıtı şeffaf hale getirir ve okuyucunun iddialarınızı bağımsız biçimde değerlendirmesine olanak tanır.

---

### Ö-2. Veri Seti Boyutunu ve Sınıf Dağılımını Raporlayın *(Öncelikli)*

Şu bilgileri bir tablo olarak yöntem bölümüne ekleyin:

| Bilgi | Eklenmesi Gereken İçerik |
|---|---|
| Ham görüntü sayısı | Toplam ve sınıf bazında |
| Artırma sonrası örnek sayısı | Eğitim / doğrulama / test ayrımıyla |
| Sınıf dağılımı | Pozitif / negatif örnek oranları |

Bu bilgi, `no_seatbelt` performans düşüşünün kaynağını nesnel olarak analiz etmeyi mümkün kılar.

---

### Ö-3. AS-2'yi Yeniden Çerçeveleyın veya Pilot Veri Toplayın

İki seçenek mevcuttur:

- **(a) Yeniden çerçeveleme:** AS-2'yi *"5G'nin teorik katkısının model hızı bulguları üzerinden dolaylı değerlendirmesi"* olarak tanımlayın ve bölüm başlığını buna göre düzenleyin. Bu, mevcut bulgularla tutarlı, dürüst bir konumlandırmadır.
- **(b) Minimal deneysel veri:** Gerçek bir 5G testi mümkün olmasa bile Raspberry Pi veya Jetson Nano üzerinde çalıştırılan pipeline'ın uçtan uca gecikme süresini (ms cinsinden) ölçüp raporlayın. Bu, iddiayı spekülasyondan çıkararak deneysel bir zemine taşır.

---

### Ö-4. Eğitim Koşullarını Standartlaştırıp Raporlayın

Adil karşılaştırma için tüm modellerin aynı koşullar altında eğitildiğini doğrulayın ve bu bilgileri yöntem bölümüne ekleyin:

| Parametre | Belirtilmesi Gereken |
|---|---|
| GPU modeli ve VRAM | Örn. NVIDIA RTX 3090, 24 GB |
| Epoch sayısı | Her model için |
| Batch size | Her model için |
| Öğrenme oranı ve optimizatör | Her model için |
| Erken durdurma kriteri | Varsa |

Farklı koşullar kullanıldıysa bunu açıkça belirtin ve karşılaştırma iddialarınızı buna göre yumuşatın.

---

### Ö-5. Inter-Annotator Agreement Ekleyin

Dört etiketleyicinin katkıda bulunduğu veri setinde **Cohen's Kappa** veya **IoU tabanlı uyum skoru** hesaplayın ve yöntem bölümüne ekleyin. Bu, bulguların güvenilirliğine yönelik somut bir güvence mekanizması işlevi görür ve metodolojik titizliğin kanıtıdır.

---

### Ö-6. Betimsel Desen Tanımını Netleştirin

Yöntem bölümünün araştırma deseni alt başlığına şu açıklamayı ekleyin:

> *"Bu çalışmada betimsel desen, karşılaştırmalı model değerlendirmesini de kapsamaktadır. Betimsel çerçeve, değişkenler üzerinde randomizasyon veya kontrol grubu ataması içermeksizin mevcut algoritmik performansın sistematik biçimde belgelenmesini ve görevler arası karşılaştırmalı yorumlanmasını ön plana almaktadır."*

---

### Ö-7. Eksik Kaynakları Tamamlayın

- "PMC, 2024" atfını tam bir kaynakla değiştirin.
- Rahutomo ve ark. (2024) için gerçek DOI'yi doldurun.
- "ve ark." biçimindeki kısaltılmış atıfları APA 7 standardına uygun tam biçime getirin.
- García-González ve ark. (2025) için DOI'yi doğrulayın.

---

### Ö-8. Hız Tahmini Modülünü Kapsama Alın veya Kapsam Dışında Bırakın

Araştırma soruları ve kapsam tanımında yer alan "hız ihlali" parametresi bulgularda hiç işlenmemektedir. İki seçenek:

- **(a)** Bu modülün bulgularını (kullanılan yöntem, elde edilen doğruluk) sunun.
- **(b)** Kapsam tanımından "hız ihlali"ni çıkarın.

Bu tutarsızlık giderilmediği sürece yapısal bir zayıflık olarak kalır.

---

### Ö-9. Güven Aralığı veya Çapraz Doğrulama Ekleyin

Yüksek mAP değerlerinin (0,986; 0,995) istatistiksel güvenilirliğini desteklemek için k-fold çapraz doğrulama veya en azından çoklu çalıştırma üzerinden standart sapma raporlaması düşünün. Bu, küçük veri setlerinden elde edilen yüksek başarım iddialarını güçlendirir.

---

## IV. Özet Değerlendirme

| Kriter | Puan | Açıklama |
|---|---|---|
| Özgünlük | ★★★★☆ | Bütünleşik yaklaşım ve αSiLU bulgusu değerli |
| Metodoloji | ★★★☆☆ | Bulgular bölümü ve veri şeffaflığı eksik |
| Literatür bağlantısı | ★★★★★ | Güçlü, kapsamlı ve tutarlı |
| Tekrarlanabilirlik | ★★☆☆☆ | Eğitim parametreleri ve veri boyutu raporlanmamış |
| Yazım kalitesi | ★★★★☆ | Akademik ve akıcı, küçük düzenlemeler yeterli |
| **Genel değerlendirme** | **Revizyon sonrasında yayınlanmaya değer** | |

---

## V. Önceliklendirme Matrisi

Sınırlılıkları güçlendirme sürecinde öneriler aşağıdaki öncelik sırasıyla ele alınmalıdır:

| Öncelik | Öneri | Etki |
|---|---|---|
| 🔴 Yüksek | Ö-1: Bulgular bölümü ekle | Yapısal bütünlük |
| 🔴 Yüksek | Ö-2: Veri seti boyutu ve sınıf dağılımı | Tekrarlanabilirlik |
| 🔴 Yüksek | Ö-7: Eksik kaynakları tamamla | Akademik dürüstlük |
| 🟡 Orta | Ö-3: AS-2'yi yeniden çerçevele | İddia-kanıt uyumu |
| 🟡 Orta | Ö-4: Eğitim koşullarını raporla | Adil karşılaştırma |
| 🟡 Orta | Ö-8: Hız tahmini tutarsızlığı | Kapsam bütünlüğü |
| 🟢 Düşük | Ö-5: Inter-annotator agreement | Metodolojik derinlik |
| 🟢 Düşük | Ö-6: Betimsel desen tanımı | Kavramsal netlik |
| 🟢 Düşük | Ö-9: Çapraz doğrulama | İstatistiksel güç |
