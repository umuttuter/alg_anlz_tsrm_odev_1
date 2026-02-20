# 🚀 Client-Side Load Balancer using Softmax Action Selection

Bu proje, Dağıtık Sistemler (Distributed Systems) mimarisinde, performansları (yanıt süreleri) zamanla değişen (non-stationary) ve gürültülü (noisy) olan K adet sunucudan oluşan bir kümeye (cluster) gelen istekleri en düşük bekleme süresiyle (latency) dağıtmayı amaçlayan bir **İstemci Taraflı Yük Dengeleyici** simülasyonudur.

Klasik *Round-Robin* veya *Random* algoritmalarının aksine, bu projede Pekiştirmeli Öğrenme (Reinforcement Learning - Multi-Armed Bandit) tabanlı **Softmax Action Selection** algoritması C dilinde sıfırdan implemente edilmiştir.

## ✨ Öne Çıkan Özellikler ve Çözülen Problemler

* **Dinamik Adaptasyon (Softmax):** Sistem, gelen isteklerin gecikmelerini ödüle ($Reward = -Latency$) çevirerek sunucuların geçmiş performanslarını (Q-değerlerini) öğrenir. Hızlı sunucuları sömürürken (Exploitation), yavaşlayan sunucuları da periyodik olarak test ederek (Exploration) sisteme dinamik olarak adapte olur.
* **Nümerik Stabilite Probleminin Çözümü (Kritik):** C dilinde Softmax algoritması hesaplanırken `exp()` fonksiyonunun sebep olduğu bellek taşması (Overflow / NaN) hatası, literatürde **Softmax Trick (Log-Sum-Exp)** olarak bilinen yöntemle çözülmüştür. Matematiksel oranlar korunarak maksimum ödül değerinin formülden çıkarılmasıyla stabilite sağlanmıştır.
* **Non-Stationary Ortam Simülasyonu:** Sunucuların yanıt süreleri sabit değildir. `rand_gaussian` fonksiyonu ile Box-Muller dönüşümü kullanılarak Normal (Gaussian) dağılıma uyan, zamanla kayma yaşayan (Random Walk) gerçekçi ağ gecikmeleri simüle edilmiştir.
* **Fiziksel Sınır Kontrolü (Edge Cases):** Rastgele yürüyüş modelinde gecikme değerlerinin matematiksel olarak negatife düşmesi durumu kontrol altına alınmış, minimum gecikme 1.0 ms'ye (clamp) sabitlenerek mantıksal tutarlılık korunmuştur.
* **Agentic Kodlama Yaklaşımı:** Bu proje, klasik kodlama yöntemleri yerine yapay zeka (LLM) ile eş-programlamalı (pair-programming) bir süreç yürütülerek geliştirilmiştir. Eski C derleyicilerinden (C89) alınan hataların çözümü ve algoritmik iyileştirmeler bu iteratif yaklaşımla sağlanmıştır.

## 🧮 Algoritma ve Matematiksel Altyapı

Softmax algoritması, her bir sunucunun seçilme olasılığını $P_i$ aşağıdaki standart formülle hesaplar:

$$P_i=\frac{e^{Q_i/\tau}}{\sum_{j=1}^{K}e^{Q_j/\tau}}$$

Ancak büyük $Q$ (tahmini ödül) değerlerinde $e^Q$ ifadesi C dilinde taşma (overflow) yaptığından, proje içerisinde formül veri setindeki en büyük ödül ($Q_{max}$) bulunarak şu şekilde stabilize edilmiştir:

$$P_i=\frac{e^{(Q_i-Q_{max})/\tau}}{\sum_{j=1}^{K}e^{(Q_j-Q_{max})/\tau}}$$

Buradaki $\tau$ (sıcaklık) parametresi sistemin yeni arayışlara girme oranını belirlerken, sistemin eski verileri unutup yeni değişimlere adapte olması sabit bir $\alpha$ (öğrenme katsayısı) ile sağlanmıştır: $Q_{yeni} = Q_{eski} + \alpha(Reward - Q_{eski})$.

## 🛠️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda derleyip çalıştırmak için herhangi bir ek kütüphaneye ihtiyaç yoktur. Standart bir C derleyicisi (GCC) yeterlidir.

1. Proje dosyalarının bulunduğu dizinde terminali (veya komut satırını) açın.
2. C dosyasını derleyin (Matematik kütüphanesini `-lm` flag'i ile bağlamayı unutmayın):
   ```bash
   gcc main.c -o load_balancer -lm