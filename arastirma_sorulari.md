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
