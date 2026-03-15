# Guia de Aplicação das Mudanças no default.html

## Resumo das mudanças no HTML

### 1. QUERY TAB — Adicionar seletor de DataMart (após o bloco de sugestões)
```html
<!-- DataMart Selector -->
<div id="datamartSelector" class="hidden flex flex-wrap gap-2 mb-3 items-center">
    <span class="text-[10px] text-fg-muted uppercase tracking-wider mr-1">DataMarts</span>
    <!-- Populated by JS renderDatamartSelector() -->
</div>
```

### 2. TABLES TAB — Upload: Adicionar campo DataMart
Após `<h2 class="text-xs font-bold text-fg-accent uppercase tracking-wider mb-3">Upload Excel</h2>`:
```html
<div class="mb-3">
    <label class="text-[10px] text-fg-muted uppercase tracking-wider block mb-1">DataMart</label>
    <div class="flex gap-2">
        <select id="uploadDatamart" class="flex-1 bg-fg-900 border border-fg-border rounded-lg px-3 py-2 text-xs font-mono focus:border-fg-accent focus:outline-none">
            <option value="default">default</option>
        </select>
        <input id="uploadDatamartNew" type="text" placeholder="ou criar novo..." class="flex-1 bg-fg-900 border border-fg-border rounded-lg px-3 py-2 text-xs font-mono focus:border-fg-accent focus:outline-none">
    </div>
</div>
```

### 3. TABLES TAB — Adicionar seção DataMart Management
Após a seção de tabelas, adicionar:
```html
<div class="mt-6 bg-fg-800 border border-fg-border rounded-xl p-5">
    <div class="flex items-center justify-between mb-3">
        <h2 class="text-xs font-bold text-fg-blue uppercase tracking-wider">DataMarts</h2>
        <button onclick="createDatamart()" class="text-xs bg-fg-blue/20 text-fg-blue border border-fg-blue/30 px-2.5 py-1 rounded-lg hover:bg-fg-blue/30 transition">+ Novo</button>
    </div>
    <div id="datamartAdminList" class="space-y-2">
        <p class="text-xs text-fg-muted">Carregando...</p>
    </div>
</div>
```

### 4. USER FORM — Adicionar checkboxes de DataMart
Após o campo "Descrição do Perfil" no `userFormPanel`:
```html
<div class="sm:col-span-2">
    <label class="text-[10px] text-fg-muted uppercase tracking-wider block mb-1">DataMarts</label>
    <div id="userFormDatamarts" class="flex flex-wrap gap-2 bg-fg-900 border border-fg-border rounded-lg p-2 max-h-[120px] overflow-y-auto">
        <!-- Populated by JS -->
    </div>
</div>
```

### 5. USER TAB — Adicionar botões Export/Import
Após "Novo Usuário" button:
```html
<button onclick="exportUsers()" class="bg-fg-green/20 text-fg-green border border-fg-green/30 px-3 py-1.5 rounded-lg text-xs font-semibold transition hover:bg-fg-green/30">Exportar Excel</button>
<button onclick="importUsers()" class="bg-fg-blue/20 text-fg-blue border border-fg-blue/30 px-3 py-1.5 rounded-lg text-xs font-semibold transition hover:bg-fg-blue/30">Importar Excel</button>
```

### 6. USER TYPE SELECT — Adicionar Root
No `userFormType`:
```html
<option value="root">Root</option>
```

### 7. SKILLS TAB — Adicionar botões Export/Import
Na seção de skills, após "Nova Skill":
```html
<button onclick="exportSkills()" class="bg-fg-green/20 text-fg-green border border-fg-green/30 px-3 py-1.5 rounded-lg text-xs font-semibold transition hover:bg-fg-green/30">Exportar Excel</button>
<button onclick="importSkills()" class="bg-fg-blue/20 text-fg-blue border border-fg-blue/30 px-3 py-1.5 rounded-lg text-xs font-semibold transition hover:bg-fg-blue/30">Importar Excel</button>
```

### 8. CHAT ACTIONS — Substituir botão Gráfico por submenu
Trocar:
```html
<button onclick="openChart()" class="...">Gráfico</button>
```
Por:
```html
<div class="relative inline-block">
    <button onclick="openChartMenu(this)" class="chart-menu-btn text-xs bg-fg-accent/15 text-fg-accent border border-fg-accent/25 px-3 py-1.5 rounded-lg hover:bg-fg-accent/25 transition font-medium">Gráfico ▾</button>
</div>
```

### 9. JAVASCRIPT — Modificar sendQuery()
Adicionar `datamart_ids` ao body do fetch:
```javascript
datamart_ids: getSelectedDatamartIds(),
```

### 10. JAVASCRIPT — Modificar uploadFile()
Pegar o DataMart do seletor:
```javascript
const dmNew = document.getElementById('uploadDatamartNew').value.trim();
const dmSelect = document.getElementById('uploadDatamart').value;
const dmName = dmNew || dmSelect || 'default';
// No fetch URL: `/api/upload?datamart_name=${encodeURIComponent(dmName)}`
```

### 11. JAVASCRIPT — Modificar loadTables()
Após carregar tabelas, chamar `loadDatamartsAdmin()` e popular `uploadDatamart`:
```javascript
// Populate upload datamart select
loadDatamartsAdmin();
```

### 12. JAVASCRIPT — Modificar DOMContentLoaded
Adicionar:
```javascript
loadDatamarts();
loadUserDatamarts();
```

### 13. JAVASCRIPT — Modificar loadUsers renderUserForm
Popular checkboxes de DataMart no formulário de edição:
```javascript
// In editUser() and showCreateUserForm():
renderUserFormDatamarts(u ? u.datamart_ids : []);
```

### 14. JAVASCRIPT — Adicionar todas as novas funções
Copiar todo o bloco de JS_ADDITIONS (DataMart state, chart submenu, export/import, etc.)
