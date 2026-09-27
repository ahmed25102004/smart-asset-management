# Smart Asset & Maintenance Management Platform

منصة سحابية ذكية وأوتوماتيكية بالكامل لإدارة وتشغيل وصيانة الأصول والمرافق والبنية التحتية.

## 🏗️ هيكلية المشروع (Flat Architecture)

```text
smart-asset-platform/
├── backend/         # السيرفر الرئيسي والـ APIs (FastAPI + PostgreSQL)
├── web-dashboard/   # لوحة تحكم الويب للمشرفين (React / Next.js)
├── mobile-app/      # تطبيق الموبايل للفنيين (Flutter - Offline First)
├── ai-service/      # خدمة الذكاء الاصطناعي والـ RAG (LangChain + Vector DB)
├── iot-simulator/   # محاكي أجهزة الـ IoT ورسائل MQTT
├── deployments/     # ملفات التشغيل التجميعي (docker-compose)
├── docs/            # توثيق النظام ومعمارية APIs والـ System Architecture
└── tests/           # اختبارات الربط الشاملة (E2E Integration Tests)
```

## 🚀 طريقة التشغيل التجميعي (Combined Sprint Demo)
من داخل مجلد `deployments`:
```bash
docker-compose up --build
```
