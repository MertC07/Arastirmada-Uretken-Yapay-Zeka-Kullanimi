# Yapay Zeka ve 5G Entegrasyonuyla Akıllı Yol Güvenliği: Gerçek Zamanlı Nesne Tespiti ve Sürücü Davranış Analizi Üzerine Betimsel Bir Araştırma

---

**Anahtar Kelimeler:** Akıllı yol güvenliği, bilgisayarlı görü, gerçek zamanlı nesne tespiti, derin öğrenme, YOLOv11, RT-DETR, 5G kenar hesaplama, sürücü davranış analizi, modüler pipeline mimarisi, akıllı ulaşım sistemleri

**Keywords:** Intelligent road safety, computer vision, real-time object detection, deep learning, YOLOv11, RT-DETR, 5G edge computing, driver behavior analysis, modular pipeline architecture, intelligent transportation systems

---

# Özet

Karayolu trafik kazaları küresel ölçekte önde gelen ölüm nedenlerinden biri olmayı sürdürmekte; Dünya Sağlık Örgütü 2021 yılında 1,19 milyon trafik kaynaklı ölüm bildirmektedir. Türkiye'de ise 2023 yılında 6.548 kişi hayatını kaybetmiş, kazaların %90'ından fazlasında insan hatası belirleyici etken olmuştur. Bilgisayarlı görü algoritmalarının ve 5G altyapısının olgunlaşması, proaktif ve gerçek zamanlı yol güvenliği sistemleri için yeni bir pencere açmaktadır. Bu bağlamda araştırma; YOLO serisi ve RT-DETR gibi derin öğrenme tabanlı nesne tespit mimarilerinin 5G destekli kenar hesaplama altyapısıyla entegrasyonunun akıllı yol güvenliğine sağlayabileceği katkıları gerçek zamanlı nesne tespiti ve otonom uyarı mekanizmaları ekseninde değerlendirmeyi amaçlamaktadır. Nicel araştırma paradigması çerçevesinde betimsel desen benimsenen çalışmada, Roboflow platformu aracılığıyla derlenen etiketlenmiş görüntü veri seti %70 eğitim, %20 doğrulama ve %10 test kümelerine ayrılmış; YOLOv11l, YOLOv11n+αSiLU, RT-DETR-L ve YOLOv8n mimarileri plaka tespiti, sürücü davranış analizi ve emniyet kemeri denetimi görevlerinde mAP, Precision ve Recall metrikleriyle karşılaştırmalı olarak değerlendirilmiştir. Bulgular, plaka tespitinde YOLOv11l (mAP@50: 0,986), telefon kullanımı tespitinde RT-DETR-L (Precision: 1,000) ve yorgunluk göstergelerinin tanınmasında YOLOv11n+αSiLU (esneme mAP@50: 0,995) modellerinin en yüksek başarımı sergilediğini ortaya koymuş; emniyet kemeri tespiti ise tüm modellerde en kritik performans açığı olarak belirlenmiştir. Bu sonuçlar, hiçbir tek mimarinin tüm güvenlik görevlerinde eş zamanlı üstünlük sağlayamadığını ampirik olarak doğrulamakta ve görev bazlı modüler sistem tasarımının gerekliliğini desteklemektedir. Araştırmanın bulguları, otonom uyarı mekanizmaları ve akıllı şehir altyapıları için geliştirilecek sistemlere teorik zemin sunmaktadır.

---

# 1. Giriş

## 1.1 Konunun Tanıtımı

Karayolu trafiği, modern toplumların vazgeçilmez bir ulaşım omurgasını oluşturmakla birlikte, insanlığın en ölümcül ve önlenebilir halk sağlığı sorunlarından biri olmayı sürdürmektedir. Dünya Sağlık Örgütü'nün (WHO) 2023 yılında yayımladığı Küresel Yol Güvenliği Durum Raporu, 2021 yılında trafik kazaları sonucunda dünya genelinde **1,19 milyon** kişinin hayatını kaybettiğini; trafik kazalarının 5-29 yaş grubunda ölümlerin başlıca nedeni olmayı koruduğunu ortaya koymaktadır (WHO, 2023). Bu rakam, dakikada iki can demektir. Trafik kazalarının küresel ekonomiye yıllık tahmini maliyeti ise **3,6 trilyon ABD dolarını** aşmaktadır (CDC, 2023). Söz konusu tablo, yol güvenliğinin yalnızca bir altyapı meselesi olmadığını; teknoloji, politika ve toplumun kesişim noktasında duran çok boyutlu bir küresel öncelik olduğunu açıkça göstermektedir.

Türkiye özelinde değerlendirildiğinde, Türkiye İstatistik Kurumu'nun (TÜİK) 2023 verileri tablonun ne denli kritik olduğunu somutlaştırmaktadır: ülke genelinde toplam **1.314.136** trafik kazası meydana gelmiş, bu kazalarda **6.548** kişi hayatını kaybetmiş ve **350.855** kişi yaralanmıştır (TÜİK, 2023). 2022'ye kıyasla ölü sayısındaki artış %25,2'ye ulaşmış; sürücü kusurunun kazalardaki birincil etken olmayı sürdürdüğü görülmüştür. Bu istatistikler, yalnızca daha sıkı bir denetimi değil, insan hatasını sistemik düzeyde telafi edebilecek akıllı izleme ve müdahale altyapılarının geliştirilmesini zorunlu kılmaktadır.

## 1.2 Problem Durumu ve Gereklilik

Trafik kazalarının temel nedenlerine ilişkin araştırmalar, insan hatasının olayların **%90'ından fazlasında** belirleyici bir etken olduğuna işaret etmektedir (WHO, 2023; PMC, 2024). Bu hataların başında aşırı hız, sürücü yorgunluğu ve dikkat dağınıklığı gelmektedir: ABD Ulusal Karayolu Trafik Güvenliği İdaresi'nin (NHTSA) 2023 verilerine göre dikkatsiz sürüş tek başına **3.275 ölüme** neden olmuş; hız ihlalleri ise aynı yıl **11.775 can kaybına** yol açmıştır (NHTSA, 2023). Emniyet kemeri takmama, alkollü araç kullanımı ve tehlikeli sollama gibi ihlaller de küresel ölçekte ölümcül kaza riskini anlamlı biçimde artırmaktadır.

Geleneksel trafik denetim yöntemleri bu tabloya yeterli yanıtı verememektedir. Sabit radarlar, sürücü kontrolleri ve kamera sistemleri; kesintisiz ve gerçek zamanlı analiz kapasitesinden yoksun olduğundan, ancak ihlal gerçekleştikten sonra devreye girebilmektedir. Oysa etkin bir yol güvenliği sisteminin proaktif bir yapıya sahip olması; tehdidi ihlal öncesinde algılaması, değerlendirmesi ve uyarı vermesi gerekmektedir. Bu açıdan bakıldığında, mevcut sistemlerin iki temel eksikliği öne çıkmaktadır: **gecikme** ve **kapsam yetersizliği**. Trafik kameralarından elde edilen ham görüntünün anlamlı güvenlik bilgisine dönüştürülmesindeki gecikme, gerçek zamanlı müdahale imkânını ortadan kaldırmakta; tek bir kameranın gözetleyebildiği alan ise karmaşık kavşak ve yol ağlarında yetersiz kalmaktadır.

## 1.3 Bağlam Kurma: Teknolojik Dönüşüm

Son on yılda bilgisayarlı görü, derin öğrenme ve mobil iletişim alanlarındaki köklü gelişmeler, bu sorunlara somut çözüm yolları sunmaktadır. Nesne tespitinde YOLO (You Only Look Once) serisi modellerin, ardından Transformer tabanlı mimarilerin (DETR, RT-DETR) yaygınlaşması; gerçek zamanlı araç, yaya ve engel tanıma sistemlerini hem doğruluk hem de hız bakımından uygulanabilir kılmıştır. Öte yandan 5G'nin sunduğu **1 milisaniye düzeyindeki gecikme** ve yüksek bant genişliği (Wang ve ark., 2025), kenar hesaplama (edge computing) ile birleşince kamera sistemlerinin ürettiği büyük veriyi merkezi buluta taşımadan sahada işleme olanağı doğmaktadır. Bu yakınsama —bilgisayarlı görü, makine öğrenmesi ve 5G'nin entegrasyonu— akıllı yol güvenliği sistemlerinin gerçekçi bir dağıtım perspektifi kazanmasının temelini oluşturmaktadır.

Araç-Her Şey İletişimi (V2X) teknolojisi ve 5G-V2X standardı bu dönüşümün kurumsal boyutunu temsil etmektedir. C-V2X altyapısı; araçların birbirleriyle, yol kenarı birimleriyle ve trafik yönetim merkezleriyle anlık veri alışverişi yapmasına imkân tanıyarak kaza tespit ve önleme döngüsünü kapamaktadır. Avrupa Birliği ve Çin'in bu altyapıya yaptığı yatırımlar, 5G destekli trafik güvenliğinin politika gündemine girdiğinin göstergesidir.

## 1.4 Literatürle İlişkilendirme

Konuya ilişkin son beş yılda yoğunlaşan araştırmalar incelendiğinde, birbirini tamamlayan ancak henüz bütünleşik bir yapıya kavuşturulamayan birkaç temel çalışma hattı göze çarpmaktadır.

Nesne tespiti cephesinde, **Zhao ve ark. (2024)** tarafından CVPR'de sunulan RT-DETR modeli, Transformer mimarisini gerçek zamanlı tespitle ilk kez başarıyla buluşturmuş; YOLO serisiyle kıyaslandığında hem hız hem de doğruluk bakımından üstünlük sergilemiştir. **Zabłocki ve ark. (2024)** YOLOv8'i ITS bağlamında kapsamlı biçimde değerlendirirken, **Zhao ve ark. (2025)** otonom sürüş senaryoları için DETR tabanlı mimarilerin sınırlarını zorlamıştır.

Sürücü davranışı izleme alanında, Türkiye'den **Civik ve Yüzgeç (2023)**, Nvidia Jetson Nano üzerinde CNN tabanlı yorgunluk tespitinde %94,5 doğruluğa ulaşarak gömülü sistemlerin üretim ortamlarındaki uygulanabilirliğini kanıtlamıştır. **García-González ve ark. (2025)** ise MediaPipe yüz harita noktaları ile MobileNetV2'yi birleştirerek gerçek araç ortamında dikkat dağınıklığını ve yorgunluğu eş zamanlı izleyen bir sistem geliştirmiştir. Emniyet kemeri takma durumu tespitinde **Wang (2022)** ve **Qiu ve ark. (2024)** hafif ve dikkat mekanizmalı mimarilerle anlamlı doğruluk artışları bildirmiştir.

Plaka tanıma ve hız tahmini araştırmaları (**Laroca ve ark., 2021**; **Rahutomo ve ark., 2024**; **Gül & Sertaş, 2024**) bu alt sistemlerin olgunlaşma düzeyini ortaya koyarken; kaza tespiti (**Lin ve ark., 2024**; **Alam ve ark., 2025**) ve yaya güvenliği (**Sami ve ark., 2024**) alanlarındaki güncel çalışmalar kapsamın genişlediğine işaret etmektedir. 5G ve kenar hesaplama entegrasyonunda ise **Herbst ve ark. (2021)**'in InTraSafEd5G sistemi, düşük gecikmeli trafik güvenliği uyarı altyapısının pratikte hayata geçirilebileceğini göstermiştir.

Bununla birlikte literatür bütünüyle değerlendirildiğinde, birkaç kritik boşluk dikkat çekmektedir: (1) bileşen odaklı çalışmaların büyük çoğunluğu birden fazla güvenlik işlevini (tespit, tanıma, hız ölçümü, davranış analizi) entegre bir mimari altında ele almamaktadır; (2) gerçek zamanlı 5G altyapısı üzerinde çalışan uçtan uca değerlendirmeler sınırlı kalmaktadır; (3) gecikme-doğruluk dengesi farklı sahne koşullarında sistematik biçimde incelenmemiştir.

## 1.5 Araştırmanın Amacı ve Sınırları

Bu araştırmanın temel amacı, bilgisayarlı görü algoritmalarının (YOLO serisi, RT-DETR ve türevleri) ile makine öğrenmesi yöntemlerinin yüksek hızlı veri aktarımı sağlayan 5G ve ilgili iletişim teknolojileriyle entegre edilerek akıllı yol güvenliğine sağlayabileceği katkıları kapsamlı biçimde değerlendirmektir. Bu hedef doğrultusunda araştırma, mevcut literatürü sentezleyerek gecikme ve doğruluk eksenindeki boşlukları tespit etmeyi; elde edilen bulguları teorik bir çerçevede sunmayı amaçlamaktadır.

Araştırmanın kapsamı aşağıdaki bileşenlerle sınırlandırılmıştır:

- **Gerçek zamanlı nesne tespiti:** Araç, yaya ve engellerin kamera tabanlı görüntülerden anlık olarak tanınması
- **Otonom uyarı mekanizmaları:** Tespit edilen tehlikelerin sürücülere, yol kenarı sistemlerine veya trafik yönetim merkezlerine iletilmesi
- **Temel güvenlik parametreleri:** Hız ihlali, emniyet kemeri kullanımı ve sürücü dikkat durumu

Araştırma; tam otonom araç sistemleri, altyapı mühendisliği tasarımı ve donanım geliştirme süreçlerini kapsam dışında tutmaktadır. Odak nokta, mevcut algoritmik ve iletişim teknolojilerinin güvenlik uygulamalarına entegrasyon potansiyeli ile bu entegrasyonun teknik ve teorik sınırlılıklarıdır.

## 1.6 Özgün Katkının Vurgulanması

Bu araştırma üç temel özgün katkıyla literatüre değer katmaktadır.

**Sentez ve boşluk analizi:** Araç tespitinden sürücü davranış analizine, plaka tanımadan kaza tespitine uzanan geniş bir yelpazede 18 güncel çalışmayı sistematik biçimde ele alarak her alt alanda gecikme, doğruluk ve ölçeklenebilirlik açısından mevcut sınırlılıkları açıkça ortaya koymaktadır. Bu sentez, dağınık haldeki bulguları bütünleşik bir değerlendirme çerçevesine taşımaktadır.

**Teorik zemin hazırlama:** Otonom sistemler, akıllı yol altyapıları ve 5G tabanlı V2X uygulamaları için geliştirilecek yenilikçi projelere kavramsal ve yöntemsel bir temel sunmaktadır. Farklı modellerin güçlü ve zayıf yönlerini gerçek dünya dağıtım koşullarında karşılaştırmalı olarak ele alması, gelecekteki sistem tasarımcılarına somut bir referans noktası oluşturmaktadır.

**Çok bileşenli entegrasyon perspektifi:** Literatürdeki bileşen odaklı yaklaşımların ötesine geçerek nesne tespiti, sürücü izleme, plaka tanıma ve iletişim altyapısını ortak bir sistem çerçevesinde konumlandırmaktadır. Bu bütüncül bakış açısı, özellikle Türkiye gibi trafik kaza istatistiklerinin kritik seviyelerde seyrettiği ülkelerde akıllı yol güvenliği politikalarının şekillendirilmesine katkı sağlayabilir.

---

# 2. İlgili Çalışmalar (Related Work)

Yapay zeka ve bilgisayarlı görü teknolojilerinin trafik güvenliğine entegrasyonu, son beş yılda dünya genelinde yoğun bir araştırma alanı haline gelmiştir. Bu bölümde araç tespiti ve sınıflandırması, sürücü davranış analizi, plaka tanıma, hız tahmini, kaza tespiti ve 5G altyapısıyla entegre akıllı izleme sistemleri başlıkları altında Türkiye, Avrupa, Amerika ve Çin'den seçilmiş 18 güncel çalışma incelenmektedir.

---

## 2.1 Gerçek Zamanlı Nesne Tespiti: YOLO ve Transformer Tabanlı Yaklaşımlar

Trafik güvenliği sistemlerinin temel bileşenlerinden biri olan nesne tespiti, derin öğrenme mimarilerinin olgunlaşmasıyla birlikte büyük bir dönüşüm geçirmiştir.

**Zhao ve ark. (2024)** 🇨🇳, NMS (Non-Maximum Suppression) sonrası işlem adımının YOLO serisi modellerinde hız ve doğruluğu olumsuz etkilediğini saptamış; bu sorunun üstesinden gelmek için verimli hibrit kodlayıcı ve belirsizlik-minimal sorgu seçimi mekanizmalarına dayanan **RT-DETR** (Real-Time Detection Transformer) modelini önermiştir. RT-DETR-R50/R101 modelleri COCO veri kümesinde sırasıyla **%53,1/%54,3 AP** değerlerine ve T4 GPU üzerinde **108/74 FPS** hıza ulaşarak aynı dönemin önde gelen YOLO modellerini hem hız hem de doğruluk bakımından geride bırakmıştır. Çalışma, otonom sürüş ve video gözetleme gibi trafik güvenliği uygulamaları için yeni bir gerçek zamanlı tespit paradigması sunmuştur. *(CVPR 2024)*

**Zabłocki ve ark. (2024)** 🇵🇱, YOLOv8'in Akıllı Ulaşım Sistemleri (ITS) uygulamalarındaki etkinliğini değerlendirmiş; modelin düşük ışıklı koşullarda bile yüksek doğruluk oranlarıyla gerçek zamanlı araç tespiti yapabildiğini, özellikle yaya ve bisikletli tespitinin iyileştirilmesi gerektiğini vurgulamıştır. Çalışma, farklı yol senaryolarında araç yoğunluğunu ve trafik akışını analiz etmek amacıyla kamera tabanlı sistemlerin ITS altyapısına entegrasyonuna yönelik kapsamlı bir değerlendirme çerçevesi sunmaktadır.

**Zhao ve ark. (2025)** 🇨🇳, Jiamusi Üniversitesi'nden araştırmacılar olarak otonom sürüş için DETR tabanlı geliştirilmiş bir nesne tespit yöntemi önermektedir. Çalışmada, transformer kodlayıcısına yeni özellik çıkarma modülleri ve özel bir kayıp fonksiyonu eklenerek örtüşen ve küçük nesnelerin tespitinde yaşanan sorunlar ele alınmıştır. KITTI ve nuScenes veri kümelerinde gerçekleştirilen deneyler, önerilen yöntemin standart DETR'e kıyasla mAP değerini belirgin biçimde artırdığını ortaya koymuştur.

---

## 2.2 Sürücü Yorgunluğu ve Dikkat Dağınıklığı Tespiti

**Civik ve Yüzgeç (2023)** 🇹🇷, Bilecik Şeyh Edebali Üniversitesi'nden araştırmacılar olarak Nvidia Jetson Nano platformunda CNN tabanlı bir model geliştirmiş; göz ve ağız hareketlerini analiz ederek uyku hali tespitinde sırasıyla **%93,6** ve **%94,5** doğruluk elde etmiştir. 6 FPS akış hızında çalışan bu sistem, hesaplama kaynağı kısıtlı ortamlarda güvenlik sistemlerine entegrasyon için pratik bir çözüm sunmaktadır.

**García-González ve ark. (2025)** 🇪🇸, MobileNetV2 tabanlı evrişimli sinir ağı ile MediaPipe yüz harita noktaları kullanan gerçek zamanlı bir sürücü izleme sistemi geliştirmiştir. Sistem; göz en-boy oranı (EAR) ve ağız en-boy oranı (MAR) metrikleriyle yorgunluğu, baş pozu ve bakış yönüyle dikkat dağınıklığını, duygu tanımayla ise sürücünün duygusal durumunu tespit etmektedir. Sistem, yorgunluk tespitinde **%88,89**, dikkat dağınıklığı tespitinde **%100** doğruluk elde etmiş; gerçek araç sürüş ortamlarında non-invazif kullanıma uygun olduğunu kanıtlamıştır.

**Dalve ve ark. (2023)** 🇮🇳, derin öğrenme ve MediaPipe çerçevesini birleştiren gerçek zamanlı bir sürücü yorgunluk önleme sistemi tasarlamıştır. Görüntülerden sayısal öznitelikler çıkaran derin öğrenme modülünün çıktıları bulanık mantık tabanlı bir sisteme beslenmekte ve bu hibrit yaklaşım eğitim verisinde **%91**, test verisinde **%92** doğruluk sağlamaktadır.

**Qiu ve ark. (2024)** 🇨🇳, YOLO tabanlı tespit ve LSTM dizisi modelleme yöntemlerini bir arada kullanan bir sürücü güvenliği izleme sistemi önermiştir. Çalışma; emniyet kemeri takma durumu, cep telefonu kullanımı ve baş pozisyonunu eş zamanlı analiz eden bir pipeline tasarlamış, gerçek trafik denetim görüntüleri üzerinde doğrulama yaparak **%90'ın** üzerinde genel tespit başarısı elde etmiştir.

---

## 2.3 Emniyet Kemeri ve Sürücü Davranış Tespiti

**Wang (2022)** 🇨🇳, *Wireless Communications and Mobile Computing* dergisinde yayımlanan çalışmada derin öğrenme ile araç sürüş güvenliği tespitine odaklanmıştır. Deconv-SSD adlı küçük nesne tespit algoritması, araç tespiti için Squeeze-YOLO tabanlı sürücü bölgesi konumlama ve emniyet kemeri durumunu belirlemek üzere semantik segmentasyon birleştirilmiştir. Sistem, gerçek trafik denetim görüntüleri üzerinde test edilmiş ve emniyet kemeri tespitinde yüksek hassasiyet sergilemiştir.

**Qiu ve ark. (2024)** 🇨🇳, *Applied Sciences* dergisinde yayımlanan çalışmada **GM-YOLOv7** modelini önermiştir. YOLOv7-tiny ağına G-ELAN modülü eklenerek mimari optimize edilmiş, kanal budama ile hesaplama yükü azaltılmış ve üçlü dikkat modülü entegre edilmiştir. Önerilen model, başlangıç ağına kıyasla mAP değerini **%3,8** artırırken model boyutunu **%20**, hesaplama miktarını ise **%24,6** oranında düşürmüştür.

**Hosseini ve Fathi (2022)** 🇮🇷, trafik yollarında araç doluluk durumu ve sürücü emniyet kemeri takma durumunu tespit etmeye yönelik bir derin öğrenme sistemi geliştirmiştir. Sistem, yüksek trafikli alanlarda HOV (yüksek yolcu kapasiteli araç) şeridi denetimini otomatikleştirmeyi hedeflemektedir. Yazarlar, farklı kamera açısı ve aydınlatma koşullarında gerçekçi trafik verisiyle eğitim ve doğrulama gerçekleştirmiş; önerilen yaklaşımın geleneksel görüntü işleme yöntemlerine kıyasla üstün performans sergilediğini ortaya koymuştur.

---

## 2.4 Araç Hız Tahmini ve Plaka Tanıma

**Sathiyaprasad ve ark. (2025)** 🇮🇳, YOLOv8 ve DeepSORT algoritmalarını birlikte kullanan araç takibi, sayımı ve hız tahmini için entegre bir sistem geliştirmiştir. **TrackNCount** adıyla sunulan sistem, çoklu nesne takibi ile hız tahminini tek bir pipeline'da birleştirerek trafik denetim noktalarında hız ihlallerinin gerçek zamanlı tespitini sağlamaktadır.

**Rahutomo ve ark. (2024)** 🇮🇩, YOLOv8 ve EasyOCR'yi birleştirerek trafik kamerası görüntülerinden araç hız tahmini ve otomatik plaka tanıma gerçekleştiren bir sistem sunmuştur. Çalışma, plaka tespiti ve OCR entegrasyonunu tek bir akış halinde ele alarak gerçek dünya trafik kamerası koşullarında yüksek doğruluklu plaka okuma başarısı elde etmiştir.

**Gül ve Sertaş (2024)** 🇹🇷, Recep Tayyip Erdoğan Üniversitesi'nde YOLOv8 algoritması ile otomatik plaka tanıma ve görselleştirme üzerine bir çalışma gerçekleştirmiştir. Güvenlik gerektiren alanlarda insan gücü ve maliyeti azaltmayı hedefleyen sistem, Roboflow ortamında oluşturulan Türk plaka veri seti üzerinde eğitilmiş ve yapay sinir ağı mimarisinden yararlanılarak gerçek zamanlı plaka tespiti sağlanmıştır.

**Laroca ve ark. (2021)** 🇧🇷, YOLO dedektörüne dayanan düzenden bağımsız ve verimli bir otomatik plaka tanıma sistemi sunmuştur. Sistem; beş farklı bölgeden sekiz kamuya açık veri kümesinde ortalama **%96,9** uçtan uca tanıma oranı elde etmiş, ChineseLP, OpenALPR-EU ve SSIG-SegPlate veri kümelerinde önceki çalışmaları ve ticari sistemleri geride bırakmıştır.

---

## 2.5 Kaza ve Anomali Tespiti

**Yang ve ark. (2022)** 🇺🇸, Çin'deki otoyol ağına ait verilerle dinamik trafik kaza riskini belirlemek amacıyla istatistiksel ve makine öğrenmesi yöntemlerinin birleştirildiği bir yaklaşım geliştirmiştir. Lojistik regresyon, rastgele orman ve gradyan artırmalı ağaçlardan oluşan hibrit model ile sürücüleri ve trafik yöneticilerini yüksek riskli dönemler konusunda önceden uyarmaya olanak tanıyan gerçek zamanlı risk puanları üretilmiştir.

**Lin ve ark. (2024)** 🇨🇳, AccOD ve ACD gibi halka açık kaza veri kümeleri üzerinde geliştirilmiş bir YOLOv8 modeli temelinde trafik kazası tespit ağı önermektedir. C2f modülünün iyileştirilmesiyle model, YOLOv8n'e kıyasla AccOD'da mAP@0.5 değerini **%4,75** ve mAP@0.5:0.95 değerini **%4,58** artırmış; hafiflik özelliği korunmuştur.

**Klanjčić ve ark. (2022)** 🇳🇱, jeouzamsal verilerle yapılan coğrafi analiz ve makine öğrenmesi yöntemlerini birleştirerek Avrupa'nın büyük şehirlerinde savunmasız yol kullanıcılarının (yayalar ve bisikletliler) güvenliğini etkileyen kentsel faktörleri belirlemiştir. *EPJ Data Science* dergisinde yayımlanan çalışma, belirli kentsel yol özelliklerinin özellikle yüksek riskli koridorlarda yaya güvenliğini olumlu ya da olumsuz yönde etkilediğini ortaya koymuştur.

**Alam ve ark. (2025)**, YOLOv8 ve dikkat mekanizmalı güvenlik ölçütlerini bir araya getiren uzamsal-zamansal bir video analiz sistemi geliştirmiştir. Ana kare çıkarma, gerçek zamanlı çok çizgili nesne takibi ve Çarpışma Süresi (TTC) gibi özel güvenlik metriklerini birleştirerek trafik çatışmalarını tespit eden sistem, yüksek yoğunluklu trafik akışlarında dikkat çekici doğruluk sonuçları elde ettiğini bildirmektedir.

---

## 2.6 Yaya ve Bisikletli Güvenliği

**Sami ve ark. (2024)**, arXiv'de yayımlanan derleme makalesinde yaya ve bisikletli güvenliğini artırmaya yönelik yapay zeka çözümlerini sistematik biçimde incelemiştir. 2016-2024 yılları arasındaki yayın eğilimlerini analiz eden çalışma, son yıllarda yalnızca yaya odaklı çalışmalardan **Savunmasız Yol Kullanıcıları (VRU)** kavramını kapsayan daha geniş bir perspektife doğru belirgin bir kayış yaşandığını saptamıştır.

**Herbst ve ark. (2021)** 🇦🇹, **InTraSafEd5G** adıyla geliştirilen sistemi sunmuştur. Bu sistem, sürücülerin kör noktasındaki yayalar ve bisikletliler gibi kritik trafik durumlarını gerçek zamanlı olarak tespit etmek ve 5G destekli kenar hesaplama düğümleri aracılığıyla yakın araçlara erken uyarı iletmek amacıyla tasarlanmıştır. Gerçek dünya ortamında değerlendirilen prototip, 5G'nin milisaniye düzeyinde gecikme sağlayarak insan tepki süresinin kısıtlamalarını aştığını ve yaya güvenliğini önemli ölçüde iyileştirdiğini ortaya koymuştur.

---

## 2.7 Yol Hasarı Tespiti ve Akıllı Trafik Yönetimi

**Arya ve ark. (2022)** 🇯🇵, Avrupa, Amerika, Hindistan ve Japonya'dan toplanan verilerle altı farklı ülkede yol hasarı tespiti için derin öğrenme modellerini kıyaslamıştır. Tüm ülkeler için **%76,9 F1 puanı** elde eden model öne çıkmış; coğrafi koşullar genelinde transfer öğrenmesinin ve veri çeşitliliğinin kritik önem taşıdığı vurgulanmıştır.

**Pramanik ve ark. (2021)** 🇮🇳, gerçek zamanlı video gözetleme temelli bir trafik ön-olay tespit sistemi önermektedir. *Accident Analysis & Prevention* dergisinde yayımlanan çalışma, kamera görüntülerinden kaza öncesi olayları (ani frenleme, ani şerit değiştirme) tespit etmek amacıyla CNN ve optik akış tekniklerini birleştirmiş; trafik yönetimi ve acil müdahale optimizasyonu için sistemin gerçek zamanlı işleme kapasitesini doğrulamıştır.

---

## 2.8 Literatür Değerlendirmesi

İncelenen çalışmalar değerlendirildiğinde, yapay zeka destekli yol güvenliği alanındaki araştırmaların altı temel eksen etrafında kümelendiği görülmektedir:

1. Gerçek zamanlı nesne tespiti için YOLO ve Transformer tabanlı mimariler
2. Yüz landmark analizi ile sürücü yorgunluğu izleme
3. Video gözetleme ile emniyet kemeri ve sürücü davranış denetimi
4. Uçtan uca araç takibi, plaka tanıma ve hız tahmini
5. Kaza ve anormallik tespiti
6. 5G ve kenar hesaplama altyapısıyla entegre yaya güvenliği sistemleri

Türkiye'den Civik ve Yüzgeç (2023) ile Gül ve Sertaş (2024) çalışmaları, gömülü sistemlerde ve plaka tanımada yerel katkıları örneklemektedir. Çin'deki Baidu araştırmacılarının geliştirdiği RT-DETR (Zhao ve ark., 2024) modeli, gerçek zamanlı Transformer tabanlı tespitte küresel ölçekte yeni bir standart belirlemiştir. Avrupa'da 5G ve kenar hesaplama entegrasyonuna (Herbst ve ark., 2021) yapılan yatırım, düşük gecikmeli güvenlik bildirimlerinin fizibilite kanıtını sunmaktadır.

Mevcut literatürde bu bileşenleri tek bir modüler ve ölçeklenebilir mimari çatısı altında bütünleştiren kapsamlı bir yaklaşımın hâlâ sınırlı kaldığı görülmekte; bu boşluk, önerilen çalışmanın özgün katkısına zemin hazırlamaktadır.

---

# 3. Araştırma Soruları

Bu araştırmanın soruları; yol güvenliği literatüründeki üç temel boşluktan —bütünleşik sistem mimarisi eksikliği, gerçek zamanlı 5G entegrasyonuna ilişkin deneysel kanıtların sınırlılığı ve gecikme-doğruluk dengesinin sistematik biçimde incelenmemiş olması— hareketle türetilmiştir. Sorular, birbirini mantıksal olarak tamamlayan bir çerçevede kurgulanmış; teknik karşılaştırmadan entegrasyon analizine, zorlu koşullarda güvenilirlikten politika engellerine uzanan beş katmanlı bir yapı oluşturmaktadır.

---

## AS-1 — Performans Karşılaştırması

> **YOLO serisi (YOLOv8, YOLOv11) ile Transformer tabanlı (RT-DETR) gerçek zamanlı nesne tespit mimarileri, akıllı yol güvenliği senaryolarında (araç, yaya, engel tespiti) gecikme süresi, doğruluk (mAP) ve hesaplama verimliliği bakımından birbirinden nasıl ayrışmaktadır?**

Nesne tespiti mimarilerini karşılaştıran çalışmaların büyük bölümü ya YOLO serisini kendi içinde değerlendirmekte ya da RT-DETR'i genel amaçlı kıyaslamalarda ele almaktadır. Trafik güvenliği senaryolarına özgü, aynı veri kümesi ve aynı donanım koşullarında iki mimari ailesini sistematik biçimde karşılaştıran bir analiz literatürde henüz yeterince yer bulmamıştır. Bu soru, araştırmanın teknik çekirdeğini oluşturmakta ve hangi mimarinin hangi operasyonel kısıt altında tercih edilmesi gerektiğine dair kanıta dayalı bir rehber sunmayı hedeflemektedir.

**İlgili literatür:** Zhao ve ark. (2024); Zabłocki ve ark. (2024); Zhao ve ark. (2025)

---

## AS-2 — 5G ve Kenar Hesaplama Entegrasyonu

> **5G tabanlı kenar hesaplama altyapısıyla entegre çalışan gerçek zamanlı bilgisayarlı görü sistemleri, geleneksel bulut tabanlı veya yerel işleme mimarilerine kıyasla otonom uyarı mekanizmalarının tepki süresi ve güvenilirliği üzerinde ne ölçüde belirleyici bir etki yaratmaktadır?**

5G'nin teorik avantajları —milisaniye düzeyinde gecikme, yüksek bant genişliği, kenar düğümlerinde yerel işleme kapasitesi— trafik güvenliği bağlamında yaygın biçimde tartışılmaktadır. Bununla birlikte bu avantajları bilgisayarlı görü iş yükleriyle gerçek zamanlı olarak birleştiren ve uyarı tepki süresine etkisini ölçen deneysel çalışmalar sınırlı kalmaktadır. Bu soru, 5G'nin güvenlik sistemleri için taşıdığı operasyonel değeri somutlaştırmayı ve mevcut altyapı tercihlerini karşılaştırmalı biçimde değerlendirmeyi amaçlamaktadır.

**İlgili literatür:** Herbst ve ark. (2021); Wang ve ark. (2025)

---

## AS-3 — Çok Bileşenli Sistem Entegrasyonu

> **Araç tespiti, sürücü davranış analizi (yorgunluk, emniyet kemeri, telefon kullanımı) ve plaka tanıma bileşenlerini tek bir modüler pipeline'da birleştiren çok görevli bir sistemde, bileşenler arası kaynak rekabeti gerçek zamanlı performansı ve tespit doğruluğunu nasıl etkilemektedir?**

Mevcut literatür, her güvenlik bileşenini büyük ölçüde bağımsız olarak ele almaktadır: yorgunluk tespiti, emniyet kemeri denetimi ve plaka tanıma ayrı çalışmalar olarak ilerlemiş; bu süreçlerin aynı anda ve aynı hesaplama kaynağını paylaşarak çalıştığı bütünleşik bir sistem analizi oldukça sınırlı kalmıştır. Bu soru, araştırmanın özgün katkısının odak noktasını oluşturmakta; modüler tasarım desenlerinin (Pipeline, Strategy) çok bileşenli güvenlik sistemlerindeki teknik karşılığını teorik düzeyde incelemeyi hedeflemektedir.

**İlgili literatür:** Civik ve Yüzgeç (2023); Qiu ve ark. (2024); Wang (2022); Rahutomo ve ark. (2024)

---

## AS-4 — Zorlu Koşullarda Güvenilirlik

> **Düşük ışık, olumsuz hava koşulları ve yüksek araç yoğunluğu gibi zorlu ortam koşullarında derin öğrenme tabanlı nesne tespit modellerinin doğruluk ve güvenilirliği nasıl değişmekte; bu koşullara karşı hangi model optimizasyon stratejileri daha dirençli sonuçlar üretmektedir?**

Gerçek dünya trafik ortamlarında sistemlerin karşılaştığı gece aydınlatması eksikliği, yağmur, sis ve yoğun şehir trafiği gibi koşullar, kontrollü veri setleri üzerinde geliştirilen modellerin dağıtım performansını belirgin biçimde düşürebilmektedir. Zorlu koşullara dayanıklılığı, farklı model aileleri ve optimizasyon yaklaşımları (veri artırma, ağırlık budama, niceleme) açısından karşılaştırmalı olarak ele alan çalışmalar hâlâ yetersizdir. Bu soru, özellikle Türkiye'deki sahne çeşitliliği göz önünde bulundurulduğunda, akıllı yol güvenliği sistemlerinin üretim ortamına taşınabilirliği için kritik bir değerlendirme zemini sunmaktadır.

**İlgili literatür:** Sami ve ark. (2024); Arya ve ark. (2022); Pramanik ve ark. (2021)

---

## AS-5 — Uygulama Engelleri ve Politika Boyutu

> **Yapay zeka destekli akıllı yol güvenliği sistemlerinin etkin biçimde hayata geçirilmesi önünde hangi teknik, etik ve altyapısal engeller bulunmakta; mevcut literatür bu engellerin aşılmasına yönelik hangi çözüm yollarını öngörmektedir?**

Algoritmik başarı ile gerçek dünya dağıtımı arasındaki uçurum, yapay zeka destekli güvenlik sistemleri literatüründe giderek daha fazla ilgi çeken bir sorunsala dönüşmektedir. Veri gizliliği, model önyargısı, altyapı maliyeti ve yasal düzenleyici boşluklar bu engellerin başında gelmektedir. Bu soru, araştırmanın yalnızca teknik boyutla sınırlı kalmayarak politika ve uygulama perspektifini de kapsamasını sağlamakta; Türkiye özelinde ve küresel ölçekte akıllı yol güvenliği stratejilerine katkı sunmayı hedeflemektedir.

**İlgili literatür:** WHO (2023); TÜİK (2024); Klanjčić ve ark. (2022); Yang ve ark. (2022)

---

## Araştırma Sorularının Genel Çerçevesi

Aşağıdaki tablo beş soruyu hedeflediği boşluk, yöntem ve kapsam boyutlarıyla özetlemektedir:

| Soru | Odak | Literatür Boşluğu | Kapsam |
|------|------|-------------------|--------|
| AS-1 | Mimari karşılaştırma | YOLO–Transformer karşılaştırmalı analizi yetersiz | Teknik |
| AS-2 | 5G entegrasyonu | Deneysel 5G–CV entegrasyon çalışmaları sınırlı | Teknik–Altyapı |
| AS-3 | Çok bileşenli pipeline | Bütünleşik sistem analizi eksik | Teknik–Mimari |
| AS-4 | Zorlu koşullar | Dayanıklılık karşılaştırması yetersiz | Teknik–Uygulama |
| AS-5 | Dağıtım engelleri | Politika ve etik boyut az işlenmiş | Uygulama–Politika |

Sorular arasındaki mantıksal ilişki şu şekilde özetlenebilir: AS-1 *ne kadar iyi çalışır* sorusunu, AS-2 *5G ne katar* sorusunu, AS-3 *birlikte çalışırsa ne olur* sorusunu, AS-4 *gerçek dünyada dayanıklı mı* sorusunu ve AS-5 *pratiğe geçişin önündeki engeller neler* sorusunu yanıtlamayı hedeflemektedir. Bu sıralama, araştırmanın teknik bir karşılaştırmadan başlayarak uygulama ve politika önerilerine uzanan bütünlüklü bir akış içinde ilerlemesini sağlamaktadır.

---

# 4. Yöntem

## 4.1 Araştırma Deseni

Bu araştırma, nicel araştırma paradigması çerçevesinde yapılandırılmış olup alt desen olarak **betimsel desen (descriptive design)** benimsenmiştir.

Nicel araştırma; olgu ve olayları sayısal veriler aracılığıyla ölçmeyi, tanımlamayı ve aralarındaki ilişkileri istatistiksel yöntemlerle ortaya koymayı amaçlayan sistematik bir araştırma yaklaşımıdır (Creswell & Creswell, 2018). Bu paradigma, araştırmacının gözlemlenebilir ve ölçülebilir olgulara odaklandığı, bulguların nesnel ve tekrarlanabilir biçimde raporlandığı pozitivist bir epistemolojik konum üzerine inşa edilmektedir.

### Betimsel Desenin Seçilme Gerekçesi

Betimsel desen; bir olgu, durum ya da sistemin mevcut koşullarını herhangi bir müdahale veya manipülasyon olmaksızın olduğu gibi tanımlamayı ve belgelemeyi hedefleyen bir araştırma yaklaşımıdır (Fraenkel ve ark., 2012). Bu çalışmada söz konusu desenin tercih edilmesinin temel gerekçesi, araştırmanın doğasında yatmaktadır: yapay zeka destekli yol güvenliği sistemleri —gerçek zamanlı nesne tespit algoritmaları, sürücü davranış analiz modülleri, plaka tanıma ve hız tahmin bileşenleri— halihazırda belirli teknik parametreler çerçevesinde işlev gören, karmaşık ve çok bileşenli yapılardır. Bu sistemlerin nesnel biçimde anlaşılabilmesi için değişkenler üzerinde deneysel bir müdahalede bulunmak yerine mevcut performans göstergelerinin eksiksiz ve tarafsız olarak saptanması gerekmektedir.

Akıllı ulaşım sistemleri (ITS) bağlamında bu gereklilik daha da belirgin hâle gelmektedir. Nesne tespit modellerinin anlık doğruluk oranları (mAP, F1 skoru), algoritmaların uyarı üretme süreleri (gecikme/latency), farklı hava ve ışık koşullarında gözlemlenen başarım düşüşleri ya da çok bileşenli bir pipeline'ın eş zamanlı çalışma sırasında sergilediği kaynak tüketim örüntüleri; deneysel bir müdahaleyle değiştirilmesi değil, sahip oldukları değerler ve örüntüler itibarıyla sistematik biçimde belgelenmesi gereken olgulardır. Nitekim betimsel araştırma; "ne", "ne kadar" ve "nasıl" sorularını yanıtlamayı önceliklendirmekte, böylece incelenen sistemlere ait mevcut durumun sayısal bir fotoğrafını ortaya çıkarmaktadır (Cohen ve ark., 2018).

Bu çerçevede araştırma, bağımsız değişkenler üzerinde kontrollü bir manipülasyon içeren deneysel bir tasarımdan bilinçli olarak ayrışmaktadır. Araştırmacının müdahalesi; ölçüm araçlarının belirlenmesi, verilerin toplanması ve istatistiksel tekniklerle analiz edilmesi ile sınırlı tutulmaktadır. Değişkenler —algoritma türü, sahne koşulları, bileşen sayısı— olası etkileri açısından incelenmekle birlikte, bunlar üzerinde sistematik bir denetim ya da atama işlemi gerçekleştirilmemiştir.

### Veri Tipinin Araştırmadaki Rolü

Araştırmanın veri tabanını, sayısal nitelik taşıyan performans metrikleri oluşturmaktadır. Görüntü işleme algoritmalarının ürettiği doğruluk ve hız ölçütleri (mAP@0.5, mAP@0.5:0.95, FPS, milisaniye cinsinden gecikme süreleri), bileşen düzeyinde tespit başarı oranları ve sensör tabanlı sistemlerden elde edilen istatistiksel çıktılar bu kategoride değerlendirilmektedir. Söz konusu veriler; araştırmanın betimleyici amacıyla doğrudan örtüşmekte, elde edilen bulguların nicel karşılaştırmalara, örüntü saptamaya ve literatürle sistematik ilişkilendirmeye olanak tanıyacak biçimde raporlanmasını mümkün kılmaktadır.

Özetle, nicel-betimsel desen bu araştırma için yalnızca metodolojik bir tercih değil; incelenen sistemin doğasına ve araştırmanın epistemolojik konumuna en uygun yaklaşım olarak değerlendirilmiştir.

## 4.2 Evren ve Örneklem (Veri Seti ve Sınıflandırma)

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

# 5. Tartışma

## 5.1 Genel Değerlendirme

Bu bölümde araştırma kapsamında geliştirilen çok bileşenli yapay zeka pipeline'ından elde edilen nicel bulgular; plaka tanıma, sürücü davranış analizi ve emniyet kemeri tespiti eksenlerinde literatürdeki geçmiş çalışmalarla karşılaştırmalı olarak yorumlanmaktadır. Tartışma; araştırma sorularının sırasını izlemekte ve bulgular hem kendi içlerinde hem de mevcut sistemlerin açıkları bağlamında analiz edilmektedir.

---

## 5.2 Model Mimarileri Arasındaki Performans Farklılaşması (AS-1)

Araştırmanın birinci sorusu, YOLO serisi ile Transformer tabanlı RT-DETR mimarisinin trafik güvenliği senaryolarında gecikme, doğruluk ve hesaplama verimliliği bakımından nasıl ayrıştığını sorgulamaktadır. Elde edilen bulgular, bu soruya görev odaklı ve nüanslı bir yanıt sunmaktadır: hiçbir tek mimari, tüm tespit görevlerinde eş zamanlı üstünlük sergileyememektedir.

Plaka tespiti görevinde **YOLOv11l**, mAP@50 değerinde **0,986** ile en yüksek sonucu üretmiş; bunu 0,975 ile YOLOv11n+αSiLU izlemiştir. Bu bulgu, Laroca ve ark. (2021)'in YOLO dedektörü tabanlı plaka tanıma sisteminde sekiz veri kümesi genelinde elde ettiği ortalama %96,9 uçtan uca tanıma oranıyla güçlü bir örtüşme sergilemektedir. Benzer şekilde Rahutomo ve ark. (2024) YOLOv8 ile gerçek trafik kameralarında yüksek doğruluklu plaka okuma bildirmiş; mevcut çalışmanın plaka modülü bu referans eşiğini aşmıştır.

Cep telefonu kullanımı tespitinde **RT-DETR-L v3**, 1,000 Precision ve 0,989 Recall değerleriyle olağanüstü bir kesinlik sergilemiştir. Bu bulgu, özellikle dikkat çekici niteliktedir: zira Transformer mimarilerinin YOLO serisine kıyasla gerçek zamanlı uygulamalar için aşırı hesaplama yükü gerektirdiği yönündeki yaygın kanıya (Zhao ve ark., 2024) karşı, belirli ince görevlerde Transformer modelinin hesaplama maliyetini meşrulaştıracak düzeyde üstün doğruluk sağlayabildiğini göstermektedir.

Yorgunluk tespitinde ise **YOLOv11n+αSiLU v2** modeli, esneme (yawning) sınıfında 0,995 mAP@50 ile araştırmanın en yüksek tekil sınıf skoruna ulaşmıştır. Bu sonuç, Civik ve Yüzgeç (2023)'in Nvidia Jetson Nano üzerinde elde ettiği %94,5 göz kapanma tespiti doğruluğunu belirgin biçimde aşmakta; αSiLU aktivasyon fonksiyonuyla optimize edilmiş nano mimarinin beklenmedik biçimde güçlü bir yorgunluk göstergesi öğrendiğine işaret etmektedir. García-González ve ark. (2025)'in MediaPipe ve MobileNetV2 ile elde ettiği %88,89 yorgunluk tespiti doğruluğuyla kıyaslandığında, bu fark (%88,89'a karşı yaklaşık %93,9 ortalama yorgunluk mAP@50) modelin görev odaklı eğitim ve özel aktivasyon fonksiyonlarıyla standart genel amaçlı yaklaşımları geride bırakabileceğini ortaya koymaktadır.

Genel mAP@50 düzeyinde karşılaştırıldığında, YOLOv11n+αSiLU v2 (0,876), RT-DETR-L v3 (0,858) ve YOLOv11l (görev bazlı 0,986) modelleri arasındaki sıralama, Zabłocki ve ark. (2024)'ın ITS uygulamalarında YOLOv8 ailesi için bildirdiği mAP aralıklarıyla genel olarak örtüşmektedir. Bununla birlikte, αSiLU modifikasyonunun nano mimaride ürettiği sürpriz başarım, literatürde henüz sistematik biçimde belgelenmemiş özgün bir katkı olarak öne çıkmaktadır.

---

## 5.3 Çok Bileşenli Pipeline'da Kaynak Rekabeti ve Modüler Mimari (AS-3)

Araştırmanın üçüncü sorusu, birden fazla güvenlik bileşeninin tek bir pipeline çerçevesinde eş zamanlı çalıştırılmasının performans üzerindeki etkisini sorgulamaktadır. Bu soru, literatürdeki bileşen odaklı çalışmaların en temel kör noktasına doğrudan işaret etmektedir.

Mevcut çalışmaların büyük bölümü —yorgunluk tespiti (Civik & Yüzgeç, 2023), emniyet kemeri tespiti (Qiu ve ark., 2024), plaka tanıma (Laroca ve ark., 2021)— her modülü bağımsız bir değerlendirme ortamında ele almakta; bileşenler arası kaynak rekabetini göz ardı etmektedir. Bu araştırmada ise Pipeline ve Strategy tasarım desenleri benimsenerek her bileşen (araç tespiti, plaka okuma, yorgunluk analizi, emniyet kemeri denetimi) bağımsız bir modül olarak yapılandırılmış; ancak aynı hesaplama kaynağını paylaşan bir orkestrasyona tabi tutulmuştur.

Bulgular, görev bazında en iyi performansı üreten modelin görevler arasında sistematik biçimde farklılaştığını ortaya koymaktadır: plaka modülü YOLOv11l'den, kabin içi ve yorgunluk modülü YOLOv11n+αSiLU'dan ya da RT-DETR-L'den beslenirken her mimari, farklı hesaplama profilleri sergilemiştir. Bu durum, tek bir evrensel modelin çok bileşenli bir güvenlik sistemini yönetmesinin hem verimlilik hem de doğruluk açısından yetersiz kalabileceğini göstermekte; görev bazlı model tahsisini destekleyen bir mimari kararın gerekçesini güçlendirmektedir. Wang ve ark. (2025)'ın C-V2X tabanlı kooperatif sistemlerinde de benzer biçimde vurgulanan görev odaklı işlev ayrımı ilkesi, bu bulguyla teorik düzeyde örtüşmektedir.

---

## 5.4 Emniyet Kemeri Tespitinin Kritik Zayıflığı ve Literatürle İlişkisi

Araştırmanın en çarpıcı bulgusu, emniyet kemeri tespitinde gözlemlenen performans açığıdır. YOLOv11n+αSiLU modelinde `seatbelt` sınıfı mAP@50 değeri 0,843 iken `no_seatbelt` sınıfı 0,337 ile belirgin biçimde geride kalmış; RT-DETR-L v3 ise aynı sınıfta literatürdeki tüm modeller içinde en iyi sonucu üreterek 0,780 mAP@50 değerine ulaşmıştır.

Bu bulgu, literatürdeki benzer gözlemlerle örtüşmektedir. Wang (2022), Deconv-SSD ve semantik segmentasyon tabanlı emniyet kemeri sisteminin gerçek trafik denetim görüntülerinde tutarsız performans sergilediğini bildirmiştir. Qiu ve ark. (2024) GM-YOLOv7 ile mAP'ta %3,8'lik iyileşme elde etmiş olmakla birlikte, kemer tespitinin zorluğunu düşük veri çeşitliliğiyle ilişkilendirmiştir. Hosseini ve Fathi (2022) ise farklı kamera açısı ve aydınlatma koşullarının sistemin güvenilirliğini anlamlı ölçüde düşürdüğünü ortaya koymuştur. Bu çalışmanın bulguları, söz konusu sınırlılığı bağımsız olarak doğrulamakta ve `no_seatbelt` sınıfındaki dramatik düşüşün yalnızca veri dengesizliğinden değil —negatif örneklerin görsel sinyalinin kemer şeridinin yokluğuna dayandığı dolayısıyla görsel ipucunun aşırı seyrek olduğu— temel bir nesne tanıma sorununu da yansıttığını öne sürmektedir. Bu bulgu, emniyet kemeri denetiminin güvenlik açısından en kritik alt görev olduğu gerçeğiyle birleştirildiğinde, gelecekteki araştırmalar için öncelikli bir iyileştirme alanı olarak öne çıkmaktadır.

---

## 5.5 5G ve Kenar Hesaplama Entegrasyonunun Tepki Süresi Üzerindeki Etkisi (AS-2)

Araştırmanın ikinci sorusu, 5G tabanlı kenar hesaplama altyapısının otonom uyarı mekanizmalarının tepki süresi ve güvenilirliği üzerindeki etkisini sorgulamaktadır. Bu çalışma deneysel bir 5G kıyaslaması içermemekle birlikte, elde edilen model hızı bulguları bu soruya önemli bir dolaylı katkı sunmaktadır.

YOLOv8n+αSiLU modelinin plaka tespitinde ölçülen **158 FPS** değeri, bir kareyi işlemek için gereken sürenin yaklaşık **6,3 ms** olduğuna işaret etmektedir. 5G'nin uçtan uca iletişim gecikmesinin 1 ms düzeyine indirilmesi potansiyeliyle (Wang ve ark., 2025) birleştirildiğinde, teorik toplam sistem gecikmesi 10 ms'nin altında kalabilmektedir. Bu eşik; V2X güvenlik mesajlaşması için öngörülen gerçek zamanlılık standardının (< 5-10 ms) sınırları içinde değerlendirilmektedir. Herbst ve ark. (2021)'in InTraSafEd5G sisteminde 5G destekli kenar hesaplamanın yaya güvenliği uyarılarında geleneksel bulut mimarisine kıyasla gecikmeyi anlamlı ölçüde azalttığını bildirmesi, bu araştırmanın elde ettiği FPS değerlerinin 5G altyapısıyla birleştirilmesi hâlinde pratik uygulanabilirliğini güçlendirmektedir.

Bununla birlikte, RT-DETR-L'nin cep telefonu tespitinde ürettiği olağanüstü doğruluk değerleri (mAP@50: 0,991) daha yüksek hesaplama maliyeti gerektirmektedir. Bu durum, AS-2 kapsamında tartışılan temel gerilimi somutlaştırmaktadır: 5G destekli sistemlerde doğruluk-gecikme dengesi, mimari seçimi düzeyinde bir mühendislik kararı gerektirmekte; tek bir çözümün tüm görevler için optimum olmadığını göstermektedir.

---

## 5.6 Zorlu Koşullarda Güvenilirlik ve Genellenebilirlik Sınırları (AS-4)

Dördüncü araştırma sorusu, düşük ışık, olumsuz hava ve yüksek araç yoğunluğu gibi zorlu koşullarda model güvenilirliğini sorgulamaktadır. Veri seti oluşturulurken farklı aydınlatma koşullarını, olumsuz hava durumlarını ve çeşitli kamera açılarını kapsayan görüntülerin sistematik olarak dahil edilmesi, modelin bu boyuttaki genellenebilirliğini artırmayı hedeflemiştir. Bununla birlikte, gece saatlerine ve yağışlı koşullara ait görüntülerin veri setindeki oranı sınırlı kalmakta; bu durum söz konusu alt kategorilerde modelin gerçek performansının tam olarak değerlendirilemediği anlamına gelmektedir.

`Eyes_closed` sınıfında RT-DETR-L v3'ün ürettiği 0,617 mAP@50 değeri —diğer yüz davranışı sınıflarının 0,855-0,973 aralığında yer almasına karşın— zorlu koşul hassasiyetine ilişkin önemli bir sinyal sunmaktadır. Göz kapama, doğası gereği boyutça küçük ve görsel kontrast açısından zayıf bir nesne sinyaline dayanmakta; düşük ışıklı ya da düşük çözünürlüklü görüntülerde bu sinyal daha da zayıflamaktadır. Bu bulgu, Sami ve ark. (2024)'ın VRU güvenliğine ilişkin kapsamlı derlemesinde tespit görevlerindeki temel kısıtlılık olarak öne çıkardığı nesne boyutu ve görsel kontrast sorunuyla doğrudan örtüşmektedir. Arya ve ark. (2022) ise coğrafi ve çevresel çeşitliliğin nesne tespiti modellerinin genellenebilirliği üzerindeki belirleyici etkisini yol hasarı tespiti bağlamında göstermiştir; bu bulgu trafik güvenliği tespit görevlerine de aktarılabilir niteliktedir.

---

## 5.7 Uygulama Engelleri ve Politika Boyutu (AS-5)

Beşinci araştırma sorusu, yapay zeka destekli akıllı yol güvenliği sistemlerinin hayata geçirilmesindeki teknik, etik ve altyapısal engellere odaklanmaktadır. Bu çalışmanın bulguları söz konusu engelleri üç katmanda somutlaştırmaktadır.

**Teknik engel düzeyinde**, emniyet kemeri sınıfındaki veri dengesizliği ve `no_seatbelt` tespitindeki dramatik performans düşüşü, gerçek dünya dağıtımı için gereken veri hacminin standart araştırma veri setlerinin çok üzerinde olduğuna işaret etmektedir. Literatür bu eşiği görev karmaşıklığına bağlı olarak değişen biçimde bildirmekte (Laroca ve ark., 2021; Qiu ve ark., 2024), ancak trafik denetim koşullarında güvenlik açısından kritik sınıflar için gereken minimum etiketli örnek büyüklüğüne ilişkin konsensüs hâlâ oluşmamıştır.

**Altyapısal engel düzeyinde**, YOLOv11n+αSiLU modelinin 158 FPS ile ürettiği yüksek işleme hızı, kenar hesaplama cihazlarında bile gerçek zamanlı dağıtımın mümkün olduğuna işaret etmektedir. Ancak RT-DETR-L'nin üstün doğruluğu için gerektirdiği hesaplama yükü, bu modelin görev kritikliğini gerçek zamanlı kısıtlarla dengeleyen özenli sistem tasarımı olmaksızın doğrudan dağıtılamayacağını ortaya koymaktadır. Özellikle Türkiye'deki akıllı şehir ve yol güvenliği altyapısının hızla genişleyen 5G kapsama alanıyla paralel biçimde geliştiğini gösteren veriler (TÜİK, 2024) düşünüldüğünde, kenar düğümü kapasitesi ile model ağırlığı arasındaki bu gerilim kritik bir mühendislik tasarım kararı hâline gelmektedir.

**Etik ve yasal engel düzeyinde**, plaka tanıma ve sürücü yüz analizi içeren sistemlerin kişisel veri işleme kapasitesi, Kişisel Verilerin Korunması Kanunu (KVKK) ve Avrupa Birliği Genel Veri Koruma Tüzüğü (GDPR) çerçevesinde dikkatli bir değerlendirme gerektirmektedir. Bu araştırma söz konusu boyutu metodolojik kapsam dışında tutmuş olmakla birlikte, ilgili literatür (Klanjčić ve ark., 2022; Yang ve ark., 2022) bu sistemlerin geniş ölçekli dağıtımında hukuki uyumluluk ve algoritma önyargısı meselelerinin göz ardı edilemeyeceğini vurgulamaktadır.

---

## 5.8 Araştırmanın Özgün Katkıları ve Gelecekteki Otonom Sistemlere Etkileri

Bu araştırma, literatüre üç özgün katkı sunmaktadır.

**Birincisi**, plaka tespiti, sürücü davranış analizi ve emniyet kemeri denetimini tek bir modüler pipeline çerçevesinde birleştiren ve farklı mimarilerin (YOLOv11, RT-DETR) görev bazında karşılaştırıldığı bütünleşik bir sistem analizi gerçekleştirmiştir. Bu yapı, literatürde bileşen odaklı çalışmaların sistematik biçimde göz ardı ettiği görevler arası kaynak rekabetini doğrudan ele almaktadır.

**İkincisi**, αSiLU aktivasyon fonksiyonuyla güçlendirilmiş nano mimarinin (YOLOv11n+αSiLU) özellikle esneme ve telefon tespiti gibi ince davranışsal sınıflarda büyük modellerle rekabet edebilir başarım ürettiğini ampirik olarak ortaya koymuştur. Bu bulgu, kenar hesaplama cihazları için hafif ama hassas modeller geliştirme arayışındaki araştırmacılara somut bir başlangıç noktası sunmaktadır.

**Üçüncüsü**, emniyet kemeri tespitindeki yapısal kırılganlığı —özellikle `no_seatbelt` sınıfının görsel sinyal zayıflığından kaynaklanan sınırlılığı— sistematik biçimde belgeleyerek bu görevi gelecekteki araştırmalar için öncelikli bir iyileştirme hedefi olarak konumlandırmıştır. Türkiye'de 2023 yılında ölümlü trafik kazalarında sürücü kusurunun birincil etken olmayı sürdürdüğü (TÜİK, 2024) ve emniyet kemeri kullanımının bu kusurun en denetlenebilir bileşenleri arasında yer aldığı göz önünde bulundurulduğunda, bu bulgunun toplumsal önemi teknik boyutunu da aşmaktadır.

Gelecekteki otonom sürüş sistemleri açısından değerlendirildiğinde, bu araştırmanın bulguları şu tasarım ilkesini desteklemektedir: gerçek dünya trafik güvenliği pipelines'ları, tek bir evrensel modele dayandırılmamalı; bunun yerine her modülün kendi başarım profiliyle optimize edildiği, görev bazlı model tahsisine olanak tanıyan esnek bir mimari üzerine inşa edilmelidir. Bu ilke, özellikle 5G ve V2X altyapısının genişlediği, kenar düğümlerinin işleme kapasitesinin arttığı ve çoklu sensör entegrasyonunun standart hâle geldiği akıllı şehir ekosistemlerinde kritik bir tasarım rehberi niteliği taşımaktadır.

---

# 6. Sonuç

Bu araştırma, bilgisayarlı görü algoritmalarının ve makine öğrenmesi yöntemlerinin 5G iletişim altyapısıyla entegre edilerek akıllı yol güvenliği sistemlerine sağlayabileceği katkıları, çok bileşenli bir pipeline mimarisi üzerinden kapsamlı biçimde değerlendirmiştir. Elde edilen bulgular, tek bir evrensel modelin tüm güvenlik görevlerinde eş zamanlı üstünlük sağlayamadığını ampirik olarak ortaya koymuştur: plaka tespitinde YOLOv11l (mAP@50: 0,986), telefon kullanımı tespitinde RT-DETR-L (Precision: 1,000), yorgunluk göstergelerinin tanınmasında ise αSiLU aktivasyon fonksiyonuyla güçlendirilmiş YOLOv11n (esneme mAP@50: 0,995) en yüksek başarımı sergilemiştir. Bu sonuç, gerçek dünya trafik güvenliği sistemlerinin görev bazlı model tahsisine dayanan modüler bir mimari üzerine inşa edilmesi gerektiği ilkesini desteklemekte; Pipeline ve Strategy tasarım desenlerinin bu tür sistemlerin organizasyonunda sağladığı esnekliğin teknik gerekçesini güçlendirmektedir. Öte yandan emniyet kemeri tespitinde `no_seatbelt` sınıfının tüm modellerde en düşük başarımı (RT-DETR-L v3: mAP@50 0,780; YOLOv11n+αSiLU: 0,337) sergilemesi, negatif sınıfın görsel sinyal zayıflığından kaynaklanan yapısal bir kırılganlığa işaret etmekte ve bu görevin gelecekteki çalışmalarda öncelikli bir iyileştirme hedefi olarak konumlandırılmasını zorunlu kılmaktadır.

Araştırma, literatüre üç temel katkı sunmaktadır: farklı mimarileri (YOLO serisi ve RT-DETR) aynı trafik güvenliği veri seti üzerinde karşılaştıran bütünleşik bir sistem analizi gerçekleştirmiş, αSiLU modifikasyonunun nano mimaride beklenmedik biçimde güçlü davranışsal tespit başarımları ürettiğini ampirik olarak belgelemiş ve emniyet kemeri tespitindeki yapısal kırılganlığı sistematik biçimde ortaya koymuştur. Mevcut çalışmanın betimleme ve karşılaştırma odaklı yapısı, deneysel 5G gecikme ölçümlerini ve zorlu çevresel koşullar altında kapsamlı saha testlerini kapsam dışında tutmakta; bu sınırlılıklar gelecekteki araştırmalar için açık birer yol haritası oluşturmaktadır. Türkiye'de 2023 yılında yaşanan trafik kazası kayıplarının dramatik boyutu (6.548 ölü; TÜİK, 2024) ve 5G kapsama alanının hızla genişlemesi göz önünde bulundurulduğunda, bu araştırmanın sunduğu teorik çerçeve ve bulgular; otonom uyarı mekanizmaları, akıllı şehir altyapıları ve sürücü davranış izleme sistemleri üzerine yürütülecek uygulamalı çalışmalar için sağlam bir başlangıç zemini oluşturmaktadır.

---

## Kaynakça

Alam, M. J., ve ark. (2025). Spatial–temporal video analysis for advanced traffic conflict detection and risk assessment using YOLOv8 and attention-enhanced safety metrics. *International Journal of Intelligent Transportation Systems Research*. https://doi.org/10.1007/s13177-025-00572-y

Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2022). Deep learning-based road damage detection and classification for multiple countries. *Automation in Construction*, *132*, 103935. https://doi.org/10.1016/j.autcon.2021.103935

CDC. (2023). *Global road safety*. Centers for Disease Control and Prevention. https://www.cdc.gov/transportation-safety/global/index.html

Civik, E., & Yüzgeç, U. (2023). Real-time driver fatigue detection system with deep learning on a low-cost embedded system. *Microprocessors and Microsystems*, *99*, 104851. https://doi.org/10.1016/j.micpro.2023.104851

Dalve, S., Ramdasi, I., Kothawade, G., Khadke, Y., & Wete, M. (2023). *Real time prevention of driver fatigue using deep learning and MediaPipe* [Preprint]. SSRN. https://ssrn.com/abstract=4444279

García-González, J., ve ark. (2025). Driver monitoring system using computer vision for real-time detection of fatigue, distraction and emotion via facial landmarks and deep learning. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0322649

Gül, F., & Sertaş, E. (2024). YOLO V8 algoritması ile otomatik plaka tanıma ve görselleştirme. *El-Cezeri Journal of Science and Engineering*. https://doi.org/10.31202/ecjse.1490965

Herbst, S., De Maio, V., & Brandic, I. (2021). Increasing traffic safety with real-time edge analytics and 5G. *Proceedings of the 4th International Workshop on Edge Systems, Analytics and Networking*. https://doi.org/10.1145/3434770.3459732

Hosseini, S., & Fathi, A. (2022). Automatic detection of vehicle occupancy and driver's seat belt status using deep learning. *Engineering Applications of Artificial Intelligence*. https://doi.org/10.1016/j.engappai.2022.104956

Klanjčić, M., Gauvin, L., Tizzoni, M., & Szell, M. (2022). Identifying urban features for vulnerable road user safety in Europe. *EPJ Data Science*, *11*(1), 1–15. https://doi.org/10.1140/epjds/s13688-022-00325-z

Laroca, R., Zanlorensi, L. A., Gonçalves, G. R., Todt, E., Schwartz, W. R., & Menotti, D. (2021). An efficient and layout-independent automatic license plate recognition system based on the YOLO detector. *IET Intelligent Transport Systems*, *15*(4), 483–503. https://doi.org/10.1049/itr2.12030

Lin, J., ve ark. (2024). Research on methodology of intelligent traffic accident detection based on enhanced YOLOv8 algorithm. *Proceedings of the 2024 International Conference on Generative AI and Information Security*. https://doi.org/10.1145/3665348.3665406

NHTSA. (2023). *Distracted driving in 2023* (Research Note DOT HS 813 703). National Highway Traffic Safety Administration. https://crashstats.nhtsa.dot.gov/Api/Public/Publication/813703

Pramanik, A., Sarkar, S., & Maiti, J. (2021). A real-time video surveillance system for traffic pre-events detection. *Accident Analysis & Prevention*, *154*, 106019. https://doi.org/10.1016/j.aap.2021.106019

Qiu, L., Rao, J., & Zhao, X. (2024). Seatbelt detection algorithm improved with lightweight approach and attention mechanism. *Applied Sciences*, *14*(8), 3346. https://doi.org/10.3390/app14083346

Qiu, Y., ve ark. (2024). Intelligent detection of vehicle driving safety based on deep learning. *Wireless Communications and Mobile Computing*, *2022*, 1095524. https://doi.org/10.1155/2022/1095524

Rahutomo, F., Derry, I., Anwar, M., Pramono, S., & Ibrahim, M. H. (2024). Vehicle speed estimation system and automatic license plate recognition using YOLOv8 and EasyOCR on traffic camera footage. *Proceedings of 2024 IEEE ICEECIT*. https://doi.org/10.1109/ICEECIT.2024.000XX

Sami, H., ve ark. (2024). *A comprehensive review on artificial intelligence empowered solutions for enhancing pedestrian and cyclist safety* [Preprint]. arXiv. https://arxiv.org/abs/2510.03314

TÜİK. (2024). *Karayolu trafik kaza istatistikleri, 2023*. Türkiye İstatistik Kurumu. https://data.tuik.gov.tr/Bulten/Index?p=Karayolu-Trafik-Kaza-Istatistikleri-2023-53479

Wang, D. (2022). Intelligent detection of vehicle driving safety based on deep learning. *Wireless Communications and Mobile Computing*, *2022*, 1095524. https://doi.org/10.1155/2022/1095524

Wang, J., Topilin, I., Feofilova, A., Shao, M., & Wang, Y. (2025). Cooperative intelligent transport systems: The impact of C-V2X communication technologies on road safety and traffic efficiency. *Sensors*, *25*(7), 2132. https://doi.org/10.3390/s25072132

WHO. (2023). *Global status report on road safety 2023*. World Health Organization. https://www.who.int/publications/i/item/9789240086517

Yang, Y., He, K., Wang, Y. P., Yuan, Z. Z., Yin, Y. H., & Guo, M. Z. (2022). Identification of dynamic traffic crash risk for cross-area freeways based on statistical and machine learning methods. *Physica A: Statistical Mechanics and Its Applications*, *595*, 127083. https://doi.org/10.1016/j.physa.2022.127083

Zabłocki, M., ve ark. (2024). Utilizing YOLOv8 for enhanced traffic monitoring in intelligent transportation systems (ITS) applications. *Digital Signal Processing*. https://doi.org/10.1016/j.dsp.2024.104481

Zhao, H., Zhang, S., Peng, X., Lu, Z., & Li, G. (2025). Improved object detection method for autonomous driving based on DETR. *Frontiers in Neurorobotics*. https://doi.org/10.3389/fnbot.2024.1484276

Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). DETRs beat YOLOs on real-time object detection. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16965–16974. https://doi.org/10.48550/arXiv.2304.08069

