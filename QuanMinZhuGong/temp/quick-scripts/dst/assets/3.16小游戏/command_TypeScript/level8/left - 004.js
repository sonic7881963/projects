
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level8/left - 004.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '89eebWIFaNKq4HsEwTz0YAZ', 'left - 004');
// 3.16小游戏/command_TypeScript/level8/left - 004.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var global___004_1 = require("./global - 004");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var mainCharacter = /** @class */ (function (_super) {
    __extends(mainCharacter, _super);
    function mainCharacter() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    mainCharacter.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    mainCharacter.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global___004_1.default.attack == true) {
            global___004_1.default.minion1_hp -= 999;
            var damage = cc.find("Canvas/bj/kuan/main_damage");
            damage.getComponent(cc.Label).string = "-999";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        global___004_1.default.attack = false;
    };
    mainCharacter.prototype.onCollisionStay = function (other) {
        global___004_1.default.attack = false;
        global___004_1.default.minion_attack = false;
    };
    mainCharacter.prototype.onCollisionExit = function (other, self) {
        cc.log("碰撞结束");
        if (global___004_1.default.minion1_hp <= 0) {
            other.node.active = false;
            global___004_1.default.attack = false;
            global___004_1.default.minion_attack = false;
            var damage = cc.find("Canvas/bj/kuan/main_damage");
            damage.getComponent(cc.Label).string = "";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
            var node1 = cc.find("Canvas/bj/d-8/c-8");
            node1.active = true;
            var node2 = cc.find("Canvas/bj/b/a");
            node2.active = true;
            self.node.active = false;
            var node3 = cc.find("Canvas/bj/kuan/tgcg");
            node3.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___004_1.default.minion1_hp, 19);
        }
    };
    mainCharacter.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    mainCharacter.prototype.update = function (dt) {
        if (global___004_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], mainCharacter.prototype, "label", void 0);
    __decorate([
        property
    ], mainCharacter.prototype, "text", void 0);
    mainCharacter = __decorate([
        ccclass
    ], mainCharacter);
    return mainCharacter;
}(cc.Component));
exports.default = mainCharacter;

cc._RF.pop();
                    }
                    if (nodeEnv) {
                        __define(__module.exports, __require, __module);
                    }
                    else {
                        __quick_compile_project__.registerModuleFunc(__filename, function () {
                            __define(__module.exports, __require, __module);
                        });
                    }
                })();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDhcXGxlZnQgLSAwMDQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7QUFDbEYsK0NBQW1DO0FBSzdCLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQTJDLGlDQUFZO0lBQXZEO1FBQUEscUVBcUZDO1FBbEZHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7SUErRTNCLENBQUM7SUF6RUcsd0JBQXdCO0lBRXhCLDhCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDM0IsQ0FBQztJQUdELFNBQVM7SUFDVCx3Q0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFDdkIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxNQUFNLEdBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBRXpCLElBQUcsc0JBQU0sQ0FBQyxNQUFNLElBQUksSUFBSSxFQUFDO1lBQ3JCLHNCQUFNLENBQUMsVUFBVSxJQUFJLEdBQUcsQ0FBQztZQUN6QixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDRCQUE0QixDQUFDLENBQUM7WUFDbkQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLE1BQU0sQ0FBQztZQUM5QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxD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LENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQzNFO1NBRUo7SUFFTCxDQUFDO0lBaEZEO1FBREMsUUFBUSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUM7Z0RBQ0k7SUFHdkI7UUFEQyxRQUFROytDQUNjO0lBTk4sYUFBYTtRQURqQyxPQUFPO09BQ2EsYUFBYSxDQXFGakM7SUFBRCxvQkFBQztDQXJGRCxBQXFGQyxDQXJGMEMsRUFBRSxDQUFDLFNBQVMsR0FxRnREO2tCQXJGb0IsYUFBYSIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuaW1wb3J0IGdsb2FibCBmcm9tIFwiLi9nbG9iYWwgLSAwMDRcIlxyXG5cclxuXHJcblxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcblxyXG5AY2NjbGFzc1xyXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBtYWluQ2hhcmFjdGVyIGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuXHJcbiAgICBAcHJvcGVydHkoY2MuTGFiZWwpXHJcbiAgICBsYWJlbDogY2MuTGFiZWwgPSBudWxsO1xyXG5cclxuICAgIEBwcm9wZXJ0eVxyXG4gICAgdGV4dDogc3RyaW5nID0gJ2hlbGxvJztcclxuICAgIG1haW5feDogbnVtYmVyO1xyXG4gICAgXHJcbiAgICBwdWJsaWMgc3RhdGljIGN1cnJlbnRfeDogbnVtYmVyO1xyXG4gICAgcHVibGljIHN0YXRpYyBjdXJyZW50X3k6IG51bWJlcjtcclxuXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuXHJcbiAgICBvbkxvYWQgKCkge1xyXG4gICAgICAgIHZhciBtYW5hZ2VyID0gY2MuZGlyZWN0b3IuZ2V0Q29sbGlzaW9uTWFuYWdlcigpO1xyXG4gICAgICAgIG1hbmFnZXIuZW5hYmxlZCA9IHRydWU7XHJcbiAgICB9XHJcbiAgICBcclxuXHJcbiAgICAvL+S6p+eUn+eisOaSnuS8muiwg+eUqFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgXHJcbiAgICAgICAgaWYoZ2xvYWJsLmF0dGFjayA9PSB0cnVlKXtcclxuICAgICAgICAgICAgZ2xvYWJsLm1pbmlvbjFfaHAgLT0gOTk5O1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL21haW5fZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIi05OTlcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UyLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCJcIjtcclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgZ2xvYWJsLmF0dGFjayA9IGZhbHNlO1xyXG4gICAgfVxyXG5cclxuXHJcbiAgICBvbkNvbGxpc2lvblN0YXkob3RoZXIpIHtcclxuICAgICAgICBnbG9hYmwuYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICAgZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgIH1cclxuXHJcbiAgICBvbkNvbGxpc2lvbkV4aXQob3RoZXIsc2VsZikge1xyXG4gICAgICAgY2MubG9nKFwi56Kw5pKe57uT5p2fXCIpO1xyXG4gICAgICAgaWYoZ2xvYWJsLm1pbmlvbjFfaHAgPD0gMCkgeyAgIFxyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgZ2xvYWJsLmF0dGFjaz0gZmFsc2U7XHJcbiAgICAgICAgZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL21haW5fZGFtYWdlXCIpO1xyXG4gICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgIGRhbWFnZTIuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIlwiO1xyXG4gICAgICAgIGxldCBub2RlMSA9IGNjLmZpbmQoXCJDYW52YXMvYmovZC04L2MtOFwiKTtcclxuICAgICAgICBub2RlMS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgIGxldCBub2RlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmovYi9hXCIpO1xyXG4gICAgICAgIG5vZGUyLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICAgc2VsZi5ub2RlLmFjdGl2ZSA9IGZhbHNlO1xyXG4gICAgICAgIGxldCBub2RlMyA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi90Z2NnXCIpO1xyXG4gICAgICAgIG5vZGUzLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoZ2xvYWJsLm1pbmlvbjFfaHAsIDE5KTtcclxuICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG4gICAgXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgIFxyXG4gICAgdGhpcy5tYWluX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgICAgIFxyXG4gICAgIFxyXG4gICAgfVxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgICBpZiAoZ2xvYWJsLmF0dGFjayA9PSAgdHJ1ZSkge1xyXG4gICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggKyAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgIFxyXG4gICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9IHRoaXMubWFpbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA+PSB0aGlzLm1haW5feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBcclxuICAgICAgICB9XHJcblxyXG4gICAgfVxyXG4gICAgXHJcbn1cclxuXHJcblxyXG4iXX0=