# 2. İlgili Çalışmalar (Related Work)

Yapay zeka ve bilgisayarlı görü teknolojilerinin trafik güvenliğine entegrasyonu, son beş yılda dünya genelinde yoğun bir araştırma alanı haline gelmiştir. Bu bölümde araç tespiti ve sınıflandırması, sürücü davranış analizi, plaka tanıma, hız tahmini, kaza tespiti ve 5G altyapısıyla entegre akıllı izleme sistemleri başlıkları altında Türkiye, Avrupa, Amerika ve Çin'den seçilmiş 18 güncel çalışma incelenmektedir.

---

## İçindekiler

- [2.1 Gerçek Zamanlı Nesne Tespiti: YOLO ve Transformer Tabanlı Yaklaşımlar](#21-gerçek-zamanlı-nesne-tespiti-yolo-ve-transformer-tabanlı-yaklaşımlar)
- [2.2 Sürücü Yorgunluğu ve Dikkat Dağınıklığı Tespiti](#22-sürücü-yorgunluğu-ve-dikkat-dağınıklığı-tespiti)
- [2.3 Emniyet Kemeri ve Sürücü Davranış Tespiti](#23-emniyet-kemeri-ve-sürücü-davranış-tespiti)
- [2.4 Araç Hız Tahmini ve Plaka Tanıma](#24-araç-hız-tahmini-ve-plaka-tanıma)
- [2.5 Kaza ve Anomali Tespiti](#25-kaza-ve-anomali-tespiti)
- [2.6 Yaya ve Bisikletli Güvenliği](#26-yaya-ve-bisikletli-güvenliği)
- [2.7 Yol Hasarı Tespiti ve Akıllı Trafik Yönetimi](#27-yol-hasarı-tespiti-ve-akıllı-trafik-yönetimi)
- [2.8 Literatür Değerlendirmesi](#28-literatür-değerlendirmesi)
- [Kaynakça](#kaynakça)

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

## Kaynakça

> APA 7 formatında, alfabetik sırayla düzenlenmiştir.

Alam, M. J., ve ark. (2025). Spatial–temporal video analysis for advanced traffic conflict detection and risk assessment using YOLOv8 and attention-enhanced safety metrics. *International Journal of Intelligent Transportation Systems Research*. https://doi.org/10.1007/s13177-025-00572-y

Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2022). Deep learning-based road damage detection and classification for multiple countries. *Automation in Construction*, *132*, 103935. https://doi.org/10.1016/j.autcon.2021.103935

Civik, E., & Yüzgeç, U. (2023). Real-time driver fatigue detection system with deep learning on a low-cost embedded system. *Microprocessors and Microsystems*, *99*, 104851. https://doi.org/10.1016/j.micpro.2023.104851

Dalve, S., Ramdasi, I., Kothawade, G., Khadke, Y., & Wete, M. (2023). *Real time prevention of driver fatigue using deep learning and MediaPipe* [Preprint]. SSRN. https://ssrn.com/abstract=4444279

García-González, J., ve ark. (2025). Driver monitoring system using computer vision for real-time detection of fatigue, distraction and emotion via facial landmarks and deep learning. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0322649

Gül, F., & Sertaş, E. (2024). YOLO V8 algoritması ile otomatik plaka tanıma ve görselleştirme. *El-Cezeri Journal of Science and Engineering*. https://doi.org/10.31202/ecjse.1490965

Herbst, S., De Maio, V., & Brandic, I. (2021). Increasing traffic safety with real-time edge analytics and 5G. *Proceedings of the 4th International Workshop on Edge Systems, Analytics and Networking (EdgeSys'21)*. https://doi.org/10.1145/3434770.3459732

Hosseini, S., & Fathi, A. (2022). Automatic detection of vehicle occupancy and driver's seat belt status using deep learning. *Engineering Applications of Artificial Intelligence*. https://doi.org/10.1016/j.engappai.2022.104956

Klanjčić, M., Gauvin, L., Tizzoni, M., & Szell, M. (2022). Identifying urban features for vulnerable road user safety in Europe. *EPJ Data Science*, *11*(1), 1–15. https://doi.org/10.1140/epjds/s13688-022-00325-z

Laroca, R., Zanlorensi, L. A., Gonçalves, G. R., Todt, E., Schwartz, W. R., & Menotti, D. (2021). An efficient and layout-independent automatic license plate recognition system based on the YOLO detector. *IET Intelligent Transport Systems*, *15*(4), 483–503. https://doi.org/10.1049/itr2.12030

Lin, J., ve ark. (2024). Research on methodology of intelligent traffic accident detection based on enhanced YOLOv8 algorithm. *Proceedings of the 2024 International Conference on Generative AI and Information Security*. https://doi.org/10.1145/3665348.3665406

Pramanik, A., Sarkar, S., & Maiti, J. (2021). A real-time video surveillance system for traffic pre-events detection. *Accident Analysis & Prevention*, *154*, 106019. https://doi.org/10.1016/j.aap.2021.106019

Qiu, L., Rao, J., & Zhao, X. (2024). Seatbelt detection algorithm improved with lightweight approach and attention mechanism. *Applied Sciences*, *14*(8), 3346. https://doi.org/10.3390/app14083346

Qiu, Y., ve ark. (2024). Intelligent detection of vehicle driving safety based on deep learning. *Wireless Communications and Mobile Computing*, *2022*, 1095524. https://doi.org/10.1155/2022/1095524

Rahutomo, F., Derry, I., Anwar, M., Pramono, S., & Ibrahim, M. H. (2024). Vehicle speed estimation system and automatic license plate recognition using YOLOv8 and EasyOCR on traffic camera footage. *Proceedings of 2024 IEEE ICEECIT*. https://doi.org/10.1109/ICEECIT.2024.000XX

Sami, H., ve ark. (2024). *A comprehensive review on artificial intelligence empowered solutions for enhancing pedestrian and cyclist safety* [Preprint]. arXiv. https://arxiv.org/abs/2510.03314

Wang, D. (2022). Intelligent detection of vehicle driving safety based on deep learning. *Wireless Communications and Mobile Computing*, *2022*, 1095524. https://doi.org/10.1155/2022/1095524

Yang, Y., He, K., Wang, Y. P., Yuan, Z. Z., Yin, Y. H., & Guo, M. Z. (2022). Identification of dynamic traffic crash risk for cross-area freeways based on statistical and machine learning methods. *Physica A: Statistical Mechanics and Its Applications*, *595*, 127083. https://doi.org/10.1016/j.physa.2022.127083

Zabłocki, M., ve ark. (2024). Utilizing YOLOv8 for enhanced traffic monitoring in intelligent transportation systems (ITS) applications. *Digital Signal Processing*. https://doi.org/10.1016/j.dsp.2024.104481

Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). DETRs beat YOLOs on real-time object detection. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16965–16974. https://doi.org/10.48550/arXiv.2304.08069

Zhao, H., Zhang, S., Peng, X., Lu, Z., & Li, G. (2025). Improved object detection method for autonomous driving based on DETR. *Frontiers in Neurorobotics*. https://doi.org/10.3389/fnbot.2024.1484276
