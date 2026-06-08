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

## Kaynakça (Tartışma Bölümü)

Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2022). Deep learning-based road damage detection and classification for multiple countries. *Automation in Construction*, *132*, 103935. https://doi.org/10.1016/j.autcon.2021.103935

Civik, E., & Yüzgeç, U. (2023). Real-time driver fatigue detection system with deep learning on a low-cost embedded system. *Microprocessors and Microsystems*, *99*, 104851. https://doi.org/10.1016/j.micpro.2023.104851

García-González, J., ve ark. (2025). Driver monitoring system using computer vision for real-time detection of fatigue, distraction and emotion via facial landmarks and deep learning. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0322649

Herbst, S., De Maio, V., & Brandic, I. (2021). Increasing traffic safety with real-time edge analytics and 5G. *Proceedings of the 4th International Workshop on Edge Systems, Analytics and Networking*. https://doi.org/10.1145/3434770.3459732

Hosseini, S., & Fathi, A. (2022). Automatic detection of vehicle occupancy and driver's seat belt status using deep learning. *Engineering Applications of Artificial Intelligence*. https://doi.org/10.1016/j.engappai.2022.104956

Klanjčić, M., Gauvin, L., Tizzoni, M., & Szell, M. (2022). Identifying urban features for vulnerable road user safety in Europe. *EPJ Data Science*, *11*(1), 1–15. https://doi.org/10.1140/epjds/s13688-022-00325-z

Laroca, R., Zanlorensi, L. A., Gonçalves, G. R., Todt, E., Schwartz, W. R., & Menotti, D. (2021). An efficient and layout-independent automatic license plate recognition system based on the YOLO detector. *IET Intelligent Transport Systems*, *15*(4), 483–503. https://doi.org/10.1049/itr2.12030

Qiu, L., Rao, J., & Zhao, X. (2024). Seatbelt detection algorithm improved with lightweight approach and attention mechanism. *Applied Sciences*, *14*(8), 3346. https://doi.org/10.3390/app14083346

Rahutomo, F., Derry, I., Anwar, M., Pramono, S., & Ibrahim, M. H. (2024). Vehicle speed estimation system and automatic license plate recognition using YOLOv8 and EasyOCR on traffic camera footage. *Proceedings of 2024 IEEE ICEECIT*. https://doi.org/10.1109/ICEECIT.2024.000XX

Sami, H., ve ark. (2024). *A comprehensive review on artificial intelligence empowered solutions for enhancing pedestrian and cyclist safety* [Preprint]. arXiv. https://arxiv.org/abs/2510.03314

TÜİK. (2024). *Karayolu trafik kaza istatistikleri, 2023*. Türkiye İstatistik Kurumu. https://data.tuik.gov.tr/Bulten/Index?p=Karayolu-Trafik-Kaza-Istatistikleri-2023-53479

Wang, D. (2022). Intelligent detection of vehicle driving safety based on deep learning. *Wireless Communications and Mobile Computing*, *2022*, 1095524. https://doi.org/10.1155/2022/1095524

Wang, J., Topilin, I., Feofilova, A., Shao, M., & Wang, Y. (2025). Cooperative intelligent transport systems: The impact of C-V2X communication technologies on road safety and traffic efficiency. *Sensors*, *25*(7), 2132. https://doi.org/10.3390/s25072132

Yang, Y., He, K., Wang, Y. P., Yuan, Z. Z., Yin, Y. H., & Guo, M. Z. (2022). Identification of dynamic traffic crash risk for cross-area freeways based on statistical and machine learning methods. *Physica A: Statistical Mechanics and Its Applications*, *595*, 127083. https://doi.org/10.1016/j.physa.2022.127083

Zabłocki, M., ve ark. (2024). Utilizing YOLOv8 for enhanced traffic monitoring in intelligent transportation systems (ITS) applications. *Digital Signal Processing*. https://doi.org/10.1016/j.dsp.2024.104481

Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). DETRs beat YOLOs on real-time object detection. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16965–16974. https://doi.org/10.48550/arXiv.2304.08069
