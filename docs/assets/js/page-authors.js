// 页面作者统计功能
document.addEventListener('DOMContentLoaded', function() {
    // 获取当前页面路径
    const currentPath = window.location.pathname;
    const fileName = getCurrentFileName(currentPath);
    
    // 创建页面作者信息容器
    createPageAuthorsContainer(fileName);
});

function getCurrentFileName(path) {
    // 从路径中提取文件名
    const segments = path.split('/').filter(segment => segment);
    if (segments.length === 0) return 'index.md';
    
    const lastSegment = segments[segments.length - 1];
    if (lastSegment === 'index.html' || lastSegment === '') {
        return 'index.md';
    }
    
    // 将 .html 转换为 .md
    return lastSegment.replace('.html', '.md');
}

async function createPageAuthorsContainer(fileName) {
    // 创建容器
    const container = document.createElement('div');
    container.className = 'page-authors-container';
    
    // 显示加载状态
    container.innerHTML = `
        <div class="page-authors-header">
            <svg class="icon" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M12,6A6,6 0 0,0 6,12A6,6 0 0,0 12,18A6,6 0 0,0 18,12A6,6 0 0,0 12,6M12,8A4,4 0 0,1 16,12A4,4 0 0,1 12,16A4,4 0 0,1 8,12A4,4 0 0,1 12,8Z"/>
            </svg>
            <span>页面 Contributor 信息</span>
        </div>
        <div style="text-align: center; padding: 1rem; color: var(--md-default-fg-color--light);">
            <em>正在加载 Contributor 信息...</em>
        </div>
    `;
    
    // 将容器添加到页面内容底部
    const content = document.querySelector('.md-content__inner');
    if (content) {
        content.appendChild(container);
    }
    
    try {
        // 获取页面元数据
        const pageMeta = await getPageMetadata(fileName);
        
        // 更新容器内容
        container.innerHTML = `
            <div class="page-authors-header">
                <svg class="icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M12,6A6,6 0 0,0 6,12A6,6 0 0,0 12,18A6,6 0 0,0 18,12A6,6 0 0,0 12,6M12,8A4,4 0 0,1 16,12A4,4 0 0,1 12,16A4,4 0 0,1 8,12A4,4 0 0,1 12,8Z"/>
                </svg>
                <span>页面 Contributor 信息</span>
            </div>
            
            <div class="page-authors-meta">
                <div class="page-authors-meta-item">
                    <svg class="icon" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M19,3H5C3.89,3 3,3.89 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19V5C21,3.89 20.1,3 19,3M19,5V19H5V5H19M17,12H7V10H17V12M15,16H7V14H15V16M17,8H7V6H17V8Z"/>
                    </svg>
                    <span>最后编辑: ${pageMeta.lastModified}</span>
                </div>
                
                <div class="page-authors-meta-item">
                    <svg class="icon" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M19,13H13V19H11V13H5V11H11V5H13V11H19V13Z"/>
                    </svg>
                    <span>创建时间: ${pageMeta.created}</span>
                </div>
            </div>
            
            <div class="page-authors-list">
                <span class="label">Page Contributors:</span>
                <div class="authors">
                    ${pageMeta.authors.map(author => `
                        <a href="${author.url}" class="page-author-item" target="_blank">
                            <img src="${author.avatar}" alt="${author.name}" class="page-author-avatar">
                            <span class="page-author-name">${author.name}</span>
                            <span class="page-author-percentage">(${author.percentage})</span>
                        </a>
                    `).join('')}
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Failed to load page contributors:', error);
        container.innerHTML = `
            <div class="page-authors-header">
                <svg class="icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M12,6A6,6 0 0,0 6,12A6,6 0 0,0 12,18A6,6 0 0,0 18,12A6,6 0 0,0 12,6M12,8A4,4 0 0,1 16,12A4,4 0 0,1 12,16A4,4 0 0,1 8,12A4,4 0 0,1 12,8Z"/>
                </svg>
                <span>页面 Contributor 信息</span>
            </div>
            <div style="text-align: center; padding: 1rem; color: var(--md-default-fg-color--light);">
                <em>无法加载 Contributor 信息</em>
            </div>
        `;
    }
}

async function getPageMetadata(fileName) {
    try {
        // 尝试从 GitHub API 获取真实的贡献者数据
        const contributors = await fetchPageContributors(fileName);
        if (contributors && contributors.length > 0) {
            return contributors;
        }
    } catch (error) {
        console.warn('Failed to fetch real contributors, using mock data:', error);
    }
    
    // // 使用模拟数据作为后备
    // const mockData = {
    //     'index.md': {
    //         lastModified: '2024年9月23日',
    //         created: '2024年9月22日',
    //         authors: [
    //             { name: 'chaochao825', percentage: '73.79%', avatar: 'https://github.com/chaochao825.png', url: 'https://github.com/chaochao825' },
    //             { name: 'contributor2', percentage: '25.24%', avatar: 'https://github.com/contributor2.png', url: 'https://github.com/contributor2' },
    //             { name: 'contributor3', percentage: '0.97%', avatar: 'https://github.com/contributor3.png', url: 'https://github.com/contributor3' }
    //         ]
    //     },
    //     'tutorial/index.md': {
    //         lastModified: '2024年9月23日',
    //         created: '2024年9月23日',
    //         authors: [
    //             { name: 'chaochao825', percentage: '85.5%', avatar: 'https://github.com/chaochao825.png', url: 'https://github.com/chaochao825' },
    //             { name: 'contributor2', percentage: '14.5%', avatar: 'https://github.com/contributor2.png', url: 'https://github.com/contributor2' }
    //         ]
    //     }
    // };
    
    // 返回对应页面的数据，如果没有则返回默认数据
    // return mockData[fileName] || {
    //     lastModified: '2024年9月23日',
    //     created: '2024年9月23日',
    //     authors: [
    //         { name: 'chaochao825', percentage: '100%', avatar: 'https://github.com/chaochao825.png', url: 'https://github.com/chaochao825' }
    //     ]
    // };
}

// 从 GitHub API 获取真实的贡献者数据
async function fetchPageContributors(fileName) {
    try {
        // 构建文件路径
        const filePath = `docs/${fileName}`;
        
        // 获取文件的提交历史
        const response = await fetch(`https://api.github.com/repos/MQ-Group/Hands_On/commits?path=${filePath}&per_page=100`);
        
        if (!response.ok) {
            throw new Error(`GitHub API error: ${response.status}`);
        }
        
        const commits = await response.json();
        
        if (commits.length === 0) {
            return null;
        }
        
        // 统计每个作者的提交次数
        const authorStats = {};
        let totalCommits = 0;
        
        commits.forEach(commit => {
            const author = commit.author || commit.committer;
            if (author) {
                const authorName = author.login || author.name;
                if (!authorStats[authorName]) {
                    authorStats[authorName] = {
                        name: authorName,
                        commits: 0,
                        avatar: author.avatar_url || `https://github.com/${authorName}.png`,
                        url: author.html_url || `https://github.com/${authorName}`
                    };
                }
                authorStats[authorName].commits++;
                totalCommits++;
            }
        });
        
        // 计算百分比并排序
        const authors = Object.values(authorStats)
            .map(author => ({
                ...author,
                percentage: ((author.commits / totalCommits) * 100).toFixed(2) + '%'
            }))
            .sort((a, b) => b.commits - a.commits);
        
        // 获取创建和最后修改时间
        const firstCommit = commits[commits.length - 1];
        const lastCommit = commits[0];
        
        const created = new Date(firstCommit.commit.author.date).toLocaleDateString('zh-CN');
        const lastModified = new Date(lastCommit.commit.author.date).toLocaleDateString('zh-CN');
        
        return {
            lastModified,
            created,
            authors
        };
        
    } catch (error) {
        console.error('Failed to fetch page contributors:', error);
        throw error;
    }
}
