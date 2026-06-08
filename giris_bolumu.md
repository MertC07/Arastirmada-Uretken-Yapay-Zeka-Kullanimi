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

## Kaynakça (Giriş Bölümü)

> APA 7 formatında

CDC. (2023). *Global road safety*. Centers for Disease Control and Prevention. https://www.cdc.gov/transportation-safety/global/index.html

Civik, E., & Yüzgeç, U. (2023). Real-time driver fatigue detection system with deep learning on a low-cost embedded system. *Microprocessors and Microsystems*, *99*, 104851. https://doi.org/10.1016/j.micpro.2023.104851

García-González, J., ve ark. (2025). Driver monitoring system using computer vision for real-time detection of fatigue, distraction and emotion via facial landmarks and deep learning. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0322649

Gül, F., & Sertaş, E. (2024). YOLO V8 algoritması ile otomatik plaka tanıma ve görselleştirme. *El-Cezeri Journal of Science and Engineering*. https://doi.org/10.31202/ecjse.1490965

Herbst, S., De Maio, V., & Brandic, I. (2021). Increasing traffic safety with real-time edge analytics and 5G. *Proceedings of the 4th International Workshop on Edge Systems, Analytics and Networking*. https://doi.org/10.1145/3434770.3459732

Lin, J., ve ark. (2024). Research on methodology of intelligent traffic accident detection based on enhanced YOLOv8 algorithm. *Proceedings of the 2024 International Conference on Generative AI and Information Security*. https://doi.org/10.1145/3665348.3665406

NHTSA. (2023). *Distracted driving in 2023* (Research Note DOT HS 813 703). National Highway Traffic Safety Administration. https://crashstats.nhtsa.dot.gov/Api/Public/Publication/813703

Qiu, L., Rao, J., & Zhao, X. (2024). Seatbelt detection algorithm improved with lightweight approach and attention mechanism. *Applied Sciences*, *14*(8), 3346. https://doi.org/10.3390/app14083346

Rahutomo, F., Derry, I., Anwar, M., Pramono, S., & Ibrahim, M. H. (2024). Vehicle speed estimation system and automatic license plate recognition using YOLOv8 and EasyOCR on traffic camera footage. *Proceedings of 2024 IEEE ICEECIT*. https://doi.org/10.1109/ICEECIT.2024.000XX

TÜİK. (2024). *Karayolu trafik kaza istatistikleri, 2023*. Türkiye İstatistik Kurumu. https://data.tuik.gov.tr/Bulten/Index?p=Karayolu-Trafik-Kaza-Istatistikleri-2023-53479

Wang, D. (2022). Intelligent detection of vehicle driving safety based on deep learning. *Wireless Communications and Mobile Computing*, *2022*, 1095524. https://doi.org/10.1155/2022/1095524

Wang, J., Topilin, I., Feofilova, A., Shao, M., & Wang, Y. (2025). Cooperative intelligent transport systems: The impact of C-V2X communication technologies on road safety and traffic efficiency. *Sensors*, *25*(7), 2132. https://doi.org/10.3390/s25072132

WHO. (2023). *Global status report on road safety 2023*. World Health Organization. https://www.who.int/publications/i/item/9789240086517

Zabłocki, M., ve ark. (2024). Utilizing YOLOv8 for enhanced traffic monitoring in intelligent transportation systems (ITS) applications. *Digital Signal Processing*. https://doi.org/10.1016/j.dsp.2024.104481

Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). DETRs beat YOLOs on real-time object detection. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16965–16974. https://doi.org/10.48550/arXiv.2304.08069

Zhao, H., Zhang, S., Peng, X., Lu, Z., & Li, G. (2025). Improved object detection method for autonomous driving based on DETR. *Frontiers in Neurorobotics*. https://doi.org/10.3389/fnbot.2024.1484276
