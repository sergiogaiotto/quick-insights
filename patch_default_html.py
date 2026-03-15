#!/usr/bin/env python3
"""
Quick Insights v2.1 — Patch default.html
Run: python patch_default_html.py <path_to_original_default.html>
Produces: default_v21.html in the same directory
"""
import sys
import re
from pathlib import Path


def patch(html: str) -> str:
    """Apply all v2.1 patches to default.html."""

    # 1. Add v21_features.js script tag before </body>
    html = html.replace(
        '</body>',
        '<script src="/static/v21_features.js"></script>\n</body>'
    )

    # 2. Add DataMart selector after suggestions div in query tab
    html = html.replace(
        '<!-- Chat area -->',
        '''<!-- DataMart Selector -->
    <div id="datamartSelector" class="hidden flex flex-wrap gap-2 mb-3 items-center"></div>

    <!-- Chat area -->'''
    )

    # 3. Add DataMart field in upload section (before drop zone)
    html = html.replace(
        '<div id="dropZone"',
        '''<div class="mb-3">
                    <label class="text-[10px] text-fg-muted uppercase tracking-wider block mb-1">DataMart</label>
                    <div class="flex gap-2">
                        <select id="uploadDatamart" class="flex-1 bg-fg-900 border border-fg-border rounded-lg px-3 py-2 text-xs font-mono focus:border-fg-accent focus:outline-none">
                            <option value="default">default</option>
                        </select>
                        <input id="uploadDatamartNew" type="text" placeholder="ou criar novo..."
                               class="flex-1 bg-fg-900 border border-fg-border rounded-lg px-3 py-2 text-xs font-mono focus:border-fg-accent focus:outline-none">
                    </div>
                </div>
                <div id="dropZone"'''
    )

    # 4. Add DataMart admin section after tablePreview div
    html = html.replace(
        '</section>\n\n<!-- ===================== SKILLS TAB',
        '''<!-- DataMart Management -->
    <div class="mt-6">
        <div class="bg-fg-800 border border-fg-border rounded-xl p-5">
            <div class="flex items-center justify-between mb-3">
                <h2 class="text-xs font-bold text-fg-blue uppercase tracking-wider">DataMarts</h2>
                <button onclick="createDatamart()" class="text-xs bg-fg-blue/20 text-fg-blue border border-fg-blue/30 px-2.5 py-1 rounded-lg hover:bg-fg-blue/30 transition">+ Novo</button>
            </div>
            <div id="datamartAdminList" class="space-y-2">
                <p class="text-xs text-fg-muted">Carregando...</p>
            </div>
        </div>
    </div>
</section>

<!-- ===================== SKILLS TAB''',
        1  # only first occurrence
    )

    # 5. Add DataMart checkboxes in user form (after profile description field)
    html = html.replace(
        '''<div class="flex gap-2 mt-3">
                <button onclick="saveUser()"''',
        '''<div class="sm:col-span-2">
                    <label class="text-[10px] text-fg-muted uppercase tracking-wider block mb-1">DataMarts</label>
                    <div id="userFormDatamarts" class="flex flex-wrap gap-2 bg-fg-900 border border-fg-border rounded-lg p-2 max-h-[120px] overflow-y-auto">
                    </div>
                </div>
            </div>
            <div class="flex gap-2 mt-3">
                <button onclick="saveUser()"'''
    )

    # 6. Add Export/Import buttons in Users tab
    html = html.replace(
        '''<button onclick="showCreateUserForm()" class="bg-fg-accent hover:bg-fg-accent-hover text-white px-3 py-1.5 rounded-lg text-xs font-semibold transition">
                Novo Usuário
            </button>''',
        '''<div class="flex gap-2">
                <button onclick="showCreateUserForm()" class="bg-fg-accent hover:bg-fg-accent-hover text-white px-3 py-1.5 rounded-lg text-xs font-semibold transition">Novo Usuário</button>
                <button onclick="exportUsers()" class="bg-fg-green/20 text-fg-green border border-fg-green/30 px-3 py-1.5 rounded-lg text-xs font-semibold transition hover:bg-fg-green/30">Exportar</button>
                <button onclick="importUsers()" class="bg-fg-blue/20 text-fg-blue border border-fg-blue/30 px-3 py-1.5 rounded-lg text-xs font-semibold transition hover:bg-fg-blue/30">Importar</button>
            </div>'''
    )

    # 7. Add Root option to user type select
    html = html.replace(
        '''<option value="user">Usuário Comum</option>
                        <option value="admin">Administrador</option>''',
        '''<option value="user">Usuário Comum</option>
                        <option value="admin">Administrador</option>
                        <option value="root">Root</option>'''
    )

    # 8. Add Export/Import buttons in Skills tab editor header
    html = html.replace(
        '''<button onclick="newSkill()" class="bg-fg-accent hover:bg-fg-accent-hover text-white px-3 py-1.5 rounded-lg text-xs font-semibold transition">
                        Nova Skill
                    </button>''',
        '''<div class="flex gap-2">
                        <button onclick="newSkill()" class="bg-fg-accent hover:bg-fg-accent-hover text-white px-3 py-1.5 rounded-lg text-xs font-semibold transition">Nova Skill</button>
                        <button onclick="exportSkills()" class="bg-fg-green/20 text-fg-green border border-fg-green/30 px-2.5 py-1.5 rounded-lg text-xs font-semibold transition hover:bg-fg-green/30">Exportar</button>
                        <button onclick="importSkills()" class="bg-fg-blue/20 text-fg-blue border border-fg-blue/30 px-2.5 py-1.5 rounded-lg text-xs font-semibold transition hover:bg-fg-blue/30">Importar</button>
                    </div>'''
    )

    # 9. Replace chart button in renderActions() with submenu trigger
    html = html.replace(
        '''<button onclick="openChart()" class="text-xs bg-fg-accent/15 text-fg-accent border border-fg-accent/25 px-3 py-1.5 rounded-lg hover:bg-fg-accent/25 transition font-medium">Gráfico</button>''',
        '''<div class="relative inline-block"><button onclick="openChartMenu(this)" class="chart-menu-btn text-xs bg-fg-accent/15 text-fg-accent border border-fg-accent/25 px-3 py-1.5 rounded-lg hover:bg-fg-accent/25 transition font-medium">Gráfico ▾</button></div>'''
    )

    # 10. Modify sendQuery to include datamart_ids
    html = html.replace(
        "skill_ids: selectedSkillIds.length > 0 ? selectedSkillIds : null,",
        "skill_ids: selectedSkillIds.length > 0 ? selectedSkillIds : null,\n                datamart_ids: getSelectedDatamartIds().length > 0 ? getSelectedDatamartIds() : null,"
    )

    # 11. Modify uploadFile to include datamart name
    html = html.replace(
        "const res = await fetch('/api/upload', { method: 'POST', body: formData });",
        """const dmNew = document.getElementById('uploadDatamartNew') ? document.getElementById('uploadDatamartNew').value.trim() : '';
        const dmSelect = document.getElementById('uploadDatamart') ? document.getElementById('uploadDatamart').value : 'default';
        const dmName = dmNew || dmSelect || 'default';
        const res = await fetch('/api/upload?datamart_name=' + encodeURIComponent(dmName), { method: 'POST', body: formData });"""
    )

    # 12. Add loadDatamarts/loadUserDatamarts to DOMContentLoaded
    html = html.replace(
        "loadActiveSkillsBadge();",
        "loadActiveSkillsBadge();\n    loadDatamarts();\n    loadUserDatamarts();"
    )

    # 13. Add loadDatamartsAdmin call in loadTables
    html = html.replace(
        "} catch (e) {\n        el.innerHTML = `<p class=\"text-xs text-red-400\">Erro: ${e.message}</p>`;\n    }\n}\n\nasync function dropTable",
        """    loadDatamartsAdmin();
    } catch (e) {
        el.innerHTML = `<p class="text-xs text-red-400">Erro: $${e.message}</p>`;
    }
}

async function dropTable"""
    )

    # 14. Modify showCreateUserForm to render DataMart checkboxes
    html = html.replace(
        "document.getElementById('userFormLogin').disabled = false;\n    document.getElementById('userFormPassword').placeholder = 'Obrigatório';\n}",
        "document.getElementById('userFormLogin').disabled = false;\n    document.getElementById('userFormPassword').placeholder = 'Obrigatório';\n    renderUserFormDatamarts([]);\n}"
    )

    # 15. Modify saveUser to include datamart_ids
    html = html.replace(
        "body: JSON.stringify({ login, user_type: userType, display_name: displayName, profile_description: profileDesc }),",
        "body: JSON.stringify({ login, user_type: userType, display_name: displayName, profile_description: profileDesc, datamart_ids: getUserFormDatamartIds() }),"
    )

    # 16. Update type labels to include root
    html = html.replace(
        "const typeLabels = { superuser: 'Super Usuário', admin: 'Administrador', user: 'Usuário' };",
        "const typeLabels = { root: 'Root', superuser: 'Super Usuário', admin: 'Administrador', user: 'Usuário' };"
    )

    # Do it again for the second occurrence
    html = html.replace(
        "const typeColors = { superuser: 'text-fg-accent', admin: 'text-fg-blue', user: 'text-fg-muted' };",
        "const typeColors = { root: 'text-red-400', superuser: 'text-fg-accent', admin: 'text-fg-blue', user: 'text-fg-muted' };"
    )

    # 17. Show users tab for root users too
    html = html.replace(
        "if (currentUser.user_type === 'superuser' || currentUser.user_type === 'admin') {",
        "if (currentUser.user_type === 'root' || currentUser.user_type === 'superuser' || currentUser.user_type === 'admin') {"
    )

    # 18. Update loadTables to also call loadDatamartsAdmin
    html = html.replace(
        "if (name === 'tables') loadTables();",
        "if (name === 'tables') { loadTables(); loadDatamartsAdmin(); }"
    )

    return html


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_default_html.py <path_to_default.html>")
        sys.exit(1)

    src = Path(sys.argv[1])
    if not src.exists():
        print(f"File not found: {src}")
        sys.exit(1)

    html = src.read_text(encoding="utf-8")
    patched = patch(html)

    dest = src.parent / "default_v21.html"
    dest.write_text(patched, encoding="utf-8")
    print(f"Patched HTML written to: {dest}")
    print(f"Original: {len(html)} chars → Patched: {len(patched)} chars")


if __name__ == "__main__":
    main()
