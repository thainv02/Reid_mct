// Lấy tất cả item từ node trước
const items = $input.all();

// Map tháng (string "01"..."12") -> Google Sheet URL
const sheetByMonth = {
  "01": "https://docs.google.com/spreadsheets/d/1xUNDrg95DkzHIMfxFtRmt5EcMjNHiO0xKxafRwu0ups/edit?gid=0#gid=0",
  "02": "https://docs.google.com/spreadsheets/d/1qu2UwuZmH_KnDzL_FktRUO4HNLclWF5tg1K253oyVXI/edit?gid=0#gid=0",
  "03": "https://docs.google.com/spreadsheets/d/1wTgr5ak410uwQppKq97ZMm_e35TvvWZtf1urtaz4NRU/edit?gid=0#gid=0",
  "04": "https://docs.google.com/spreadsheets/d/1P_wxaDoz3YMz0FmQc2zneXqBrJ0DnIPRIDQNkEO7Hi8/edit?gid=0#gid=0",
  "05": "https://docs.google.com/spreadsheets/d/1WKDulY1tJMeVM8iFZu9EFrKNN5l85mE_elR3xN29PwI/edit?gid=0#gid=0",
  "06": "https://docs.google.com/spreadsheets/d/1xu6DtLMCCakXAjuosuLHo3JVViTN3DziU4GxHD1jH9A/edit?gid=0#gid=0",
  "07": "https://docs.google.com/spreadsheets/d/1Mp-31CfvZ3vEEA14upynwHMGPjXD5QJw0ETaQKwmHCo/edit?gid=0#gid=0",
  "08": "https://docs.google.com/spreadsheets/d/1pLXZk86Z0u8_sBhJdzpDYH8Dyd2Jf_O_tC_dhNKqIlk/edit?gid=0#gid=0",
  "09": "https://docs.google.com/spreadsheets/d/1ohaRfS8n8sqKlXhp_IBqqP5ZBoSBYEJafFjJeBnBW_k/edit?gid=0#gid=0",
  "10": "https://docs.google.com/spreadsheets/d/1-vSCeIAeQ31hPnafwDY64TL8M7ldm1tIO1detgK3AZU/edit?gid=0#gid=0",
  "11": "https://docs.google.com/spreadsheets/d/1nNEMmjTb62Od-1xcZ3lkfDw2dJJOH6hcskWMkNGi2NY/edit?gid=0#gid=0",
  "12": "https://docs.google.com/spreadsheets/d/1GUhioggvjCdMEeTrwSWyIVItKiAJj6xmXx_TiIhDksw/edit?gid=0#gid=0",
};

return items.map(item => {
  const month = item.json.month; // "01", "02", ...

  // Lấy URL theo tháng, nếu chưa cấu hình thì cho rỗng hoặc giá trị mặc định
  const sheetUrl = sheetByMonth[month] || "";

  return {
    json: {
      ...item.json,   // giữ lại month, year, vietnamTime từ node trước
      sheetUrl        // thêm trường sheetUrl để node sau dùng
    }
  };
});
