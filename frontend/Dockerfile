# --- Build stage ---
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# --- Run stage ---
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
RUN adduser -D -u 1001 nextjs
# เอาเฉพาะไฟล์ที่ต้องใช้ตอนรัน
COPY --from=builder /app/package*.json ./
RUN npm ci --omit=dev
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
# ถ้าคุณมีไฟล์ config อื่นที่ต้องใช้ตอนรันให้ COPY มาด้วย (เช่น next.config.js ถ้าจำเป็น)
EXPOSE 3000
USER nextjs
CMD ["npm","run","start"]
